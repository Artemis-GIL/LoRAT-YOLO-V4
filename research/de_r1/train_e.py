"""Train only the spatial residual on the fixed B checkpoint and Train split."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import (
    OLD, RUN, B_WEIGHTS, read, save, sha, setup, verify, load_adapter,
    freeze_stage, baseline_files, verify_baseline,
)
from .spatial_e import install_spatial
from research.bc_r1.data import PairDataset, collate
from research.bc_r1.train_b import criterion, batch_to_cuda


def frozen_model_digest(model):
    """Hash frozen parameter values and buffers, including the original head."""
    digest = hashlib.sha256()
    tensors = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    tensors += list(model.named_buffers())
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_branch(branch, path):
    from safetensors.torch import save_file
    path = Path(path)
    tmp = path.with_suffix('.partial.safetensors')
    save_file({n: p.detach().cpu().contiguous() for n, p in branch.state_dict().items()}, str(tmp))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    protocol = read(RUN / 'PROTOCOL.json')
    cfg = protocol['E']
    seed = protocol.get('seed', 20261004)
    setup(seed)
    folder = RUN / ('E_smoke' if args.smoke else 'E')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_baseline()
        sources = [Path(__file__), Path(__file__).with_name('spatial_e.py'),
                   Path(__file__).with_name('common.py'), RUN / 'PROTOCOL.json',
                   *baseline_files()]
        frozen = freeze_stage(folder.name, sources)
        if (folder / 'COMPLETE.json').exists():
            complete = read(folder / 'COMPLETE.json')
            verify(frozen)
            assert sha(folder / 'final.safetensors') == complete['sha256']
            print('E_ALREADY_COMPLETE', folder, flush=True)
            return

        adapter = load_adapter(folder / 'MODEL.json', B_WEIGHTS)
        model = adapter.model
        branch = install_spatial(adapter)
        model.eval()
        branch.train()
        params = list(branch.parameters())
        assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
        save(folder / 'SPATIAL_MODEL.json', dict(
            baseline_checkpoint=str(B_WEIGHTS), baseline_sha256=sha(B_WEIGHTS),
            insertion='after native fused search tokens, before the unchanged frozen MLP head',
            search_token_grid=list(model.x_size), hidden=64,
            trainable_parameters=sum(p.numel() for p in params),
            trainable_names=[name for name, p in model.named_parameters() if p.requires_grad],
            frozen_B=True, zero_initialized_output_projection=True,
            loss='unchanged official IoU-aware BCE + GIoU, weights 1:1',
        ))
        loss_fn = criterion()
        groups = [dict(params=[p for p in params if p.ndim > 1], weight_decay=cfg['weight_decay']),
                  dict(params=[p for p in params if p.ndim <= 1], weight_decay=0.)]
        opt = torch.optim.AdamW(groups, lr=cfg['learning_rate'])
        epochs = cfg['epochs']
        accumulation = 1 if args.smoke else cfg['accumulation']
        batches_per_epoch, remainder = divmod(cfg['samples_per_epoch'], cfg['microbatch'])
        assert not remainder and batches_per_epoch % cfg['accumulation'] == 0
        total_updates = epochs * batches_per_epoch // cfg['accumulation']
        start_epoch = updates = backwards = 0
        elapsed_previous = 0.
        if (folder / 'resume.pt').exists() and not args.smoke:
            state = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
            assert state['baseline_sha256'] == sha(B_WEIGHTS)
            branch.load_state_dict(state['branch'], strict=True)
            opt.load_state_dict(state['optimizer'])
            start_epoch, updates = state['next_epoch'], state['updates']
            backwards = state['real_backward_steps']
            elapsed_previous = state['elapsed_seconds']
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        initial_digest = frozen_model_digest(model)
        before = {n: p.detach().cpu().clone() for n, p in branch.named_parameters()} if args.smoke else None
        cases = [c for c in read(OLD / 'PROTOCOL.json')['cases'] if c['split'] == 'train']
        assert len(cases) == 1268
        started = time.time()
        actual_steps = 0
        upstream_gradient_seen = False
        frozen_gradients_absent = True
        torch.cuda.reset_peak_memory_stats()
        with (folder / 'train.jsonl').open('a', buffering=1) as trace:
            for epoch in range(start_epoch, 1 if args.smoke else epochs):
                dataset = PairDataset(cases, cfg, epoch, seed)
                workers = 0 if args.smoke else cfg.get('workers', 4)
                loader = DataLoader(
                    dataset, batch_size=cfg['microbatch'], num_workers=workers,
                    collate_fn=collate, pin_memory=True, shuffle=False,
                    **(dict(multiprocessing_context='spawn', prefetch_factor=2) if workers else {}),
                )
                opt.zero_grad(set_to_none=True)
                running = 0.
                seen = 0
                for index, batch in enumerate(loader):
                    batch = batch_to_cuda(batch)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(batch['z'], batch['x'], batch['mask'])
                    loss = loss_fn(output, batch['target']).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite E loss: epoch={epoch + 1}, batch={index + 1}')
                    (loss / accumulation).backward()
                    backwards += 1
                    actual_steps += 1
                    running += float(loss.detach())
                    seen += 1
                    if args.smoke:
                        frozen_gradients_absent &= all(p.grad is None for p in model.parameters() if not p.requires_grad)
                        upstream_gradient_seen |= any(
                            p.grad is not None and bool(p.grad.detach().abs().sum() > 0)
                            for name, p in branch.named_parameters() if not name.startswith('up.')
                        )
                    if (index + 1) % accumulation == 0:
                        grad = torch.nn.utils.clip_grad_norm_(params, 1.)
                        if not torch.isfinite(grad):
                            raise RuntimeError('Nonfinite E gradient')
                        if updates < cfg['warmup_steps']:
                            lr = cfg['learning_rate'] * (updates + 1) / cfg['warmup_steps']
                        else:
                            frac = (updates - cfg['warmup_steps']) / max(1, total_updates - cfg['warmup_steps'])
                            lr = cfg['minimum_learning_rate'] + .5 * (
                                cfg['learning_rate'] - cfg['minimum_learning_rate']
                            ) * (1 + math.cos(math.pi * frac))
                        for group in opt.param_groups:
                            group['lr'] = lr
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        updates += 1
                        record = dict(epoch=epoch + 1, batch=index + 1, updates=updates,
                                      real_backward_steps=backwards, loss=float(loss.detach()),
                                      learning_rate=lr, grad_norm=float(grad),
                                      elapsed_seconds=elapsed_previous + time.time() - started,
                                      peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30)
                        trace.write(json.dumps(record) + '\n')
                        if updates % 10 == 0 or args.smoke:
                            save(folder / 'PROGRESS.json', record)
                            print('E_TRAIN', json.dumps(record), flush=True)
                    if args.smoke and actual_steps == 2:
                        break
                if args.smoke:
                    break
                save_branch(branch, folder / f'epoch_{epoch + 1:02d}.safetensors')
                state = dict(branch={n: p.detach().cpu().clone() for n, p in branch.state_dict().items()},
                             optimizer=opt.state_dict(), next_epoch=epoch + 1, updates=updates,
                             real_backward_steps=backwards, baseline_sha256=sha(B_WEIGHTS),
                             elapsed_seconds=elapsed_previous + time.time() - started,
                             torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(state, folder / 'resume.partial.pt')
                os.replace(folder / 'resume.partial.pt', folder / 'resume.pt')
                save(folder / f'epoch_{epoch + 1:02d}.json',
                     dict(epoch=epoch + 1, mean_loss=running / seen, updates=updates,
                          real_backward_steps=backwards))
                print('E_EPOCH_DONE', epoch + 1, flush=True)

        final_digest = frozen_model_digest(model)
        assert final_digest == initial_digest, 'Frozen B parameter/buffer values changed'
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        verify(frozen)
        verify_baseline()
        if args.smoke:
            changed = [name for name, p in branch.named_parameters() if not torch.equal(p.detach().cpu(), before[name])]
            assert actual_steps == 2 and updates == 2 and changed
            assert frozen_gradients_absent and upstream_gradient_seen
        else:
            assert updates == total_updates
            assert backwards == epochs * batches_per_epoch
        save_branch(branch, folder / 'final.safetensors')
        complete = dict(status='PASS' if args.smoke else 'COMPLETE', epochs=0 if args.smoke else epochs,
                        updates=updates, real_backward_steps=backwards,
                        trainable_parameters=sum(p.numel() for p in params),
                        checkpoint=str(folder / 'final.safetensors'), sha256=sha(folder / 'final.safetensors'),
                        baseline_sha256=sha(B_WEIGHTS), frozen_model_digest=final_digest,
                        frozen_B_unchanged=True, frozen_B_gradients_absent=True,
                        elapsed_seconds=elapsed_previous + time.time() - started,
                        peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30)
        if args.smoke:
            complete.update(changed_branch_tensors=len(changed), upstream_branch_gradient_seen=upstream_gradient_seen)
        save(folder / 'COMPLETE.json', complete)
        save(folder / 'PROGRESS.json', complete)
        print('E_COMPLETE', folder, flush=True)


if __name__ == '__main__':
    main()
