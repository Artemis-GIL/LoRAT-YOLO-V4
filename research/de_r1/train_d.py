"""Train matched static/temporal action heads from frozen B causal rollouts."""
import argparse,fcntl,time,os
import torch
from safetensors.torch import save_file
from .common import *
from .data_d import Samples,collate
from .model_d import ActionValue,action_loss

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['D0','D'],required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    protocol=read(RUN/'PROTOCOL.json');cfg=protocol['D'];setup(protocol['seed'])
    folder=RUN/(args.variant+('_smoke' if args.smoke else ''));folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_baseline()
        data=RUN/'D_data';assert sha(data/'SUMMARY.json')==read(data/'COMPLETE.json')['summary_sha256']
        summary=read(data/'SUMMARY.json');verify(summary['label_files']);verify(summary['source_samples'])
        files=[Path(__file__),Path(__file__).with_name('model_d.py'),Path(__file__).with_name('data_d.py'),
            Path(__file__).with_name('history.py'),Path(__file__).with_name('common.py'),RUN/'PROTOCOL.json',data/'SUMMARY.json',*baseline_files()]
        files+=list(map(Path,summary['label_files']))
        frozen=freeze_stage(folder.name,files)
        if (folder/'COMPLETE.json').exists():
            assert sha(folder/'final.safetensors')==read(folder/'COMPLETE.json')['checkpoint_sha256'];return
        dataset=Samples(data);model=ActionValue(temporal=args.variant=='D').cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        rng=np.random.default_rng(protocol['seed']);start_epoch=0;updates=0;elapsed_prior=0.
        if (folder/'resume.pt').exists() and not args.smoke:
            state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
            rng.bit_generator.state=state['rng'];start_epoch=state['next_epoch'];updates=state['updates'];elapsed_prior=state['elapsed_seconds']
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters()} if args.smoke else None
        start=time.time()
        with (folder/'train.jsonl').open('a',buffering=1) as trace:
            for epoch in range(start_epoch,1 if args.smoke else cfg['epochs']):
                losses=[]
                for step in range(2 if args.smoke else cfg['steps_per_epoch']):
                    ids=rng.integers(0,len(dataset),cfg['batch_size'])
                    batch={k:v.cuda() for k,v in collate([dataset.get(i) for i in ids]).items()}
                    inputs={k:batch[k] for k in ['features','metadata','geometry','history_valid','baseline_index','padding']}
                    output=model(**inputs)
                    loss,parts=action_loss(output,batch['quality'],batch['target_delta'],batch['padding'],batch['baseline_index'])
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad)
                    optimizer.step();updates+=1;losses.append(float(loss.detach()))
                    record=dict(epoch=epoch+1,step=step+1,updates=updates,loss=float(loss.detach()),grad_norm=float(grad),
                        **{k:float(v) for k,v in parts.items()},elapsed_seconds=elapsed_prior+time.time()-start)
                    trace.write(json.dumps(record)+'\n')
                    if updates%100==0 or args.smoke:save(folder/'PROGRESS.json',record);print(args.variant+'_TRAIN',json.dumps(record),flush=True)
                save(folder/f'epoch_{epoch+1:02d}.json',dict(epoch=epoch+1,mean_loss=float(np.mean(losses)),updates=updates))
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(folder/f'epoch_{epoch+1:02d}.safetensors'))
                state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                    rng=rng.bit_generator.state,next_epoch=epoch+1,updates=updates,elapsed_seconds=elapsed_prior+time.time()-start)
                torch.save(state,folder/'resume.partial.pt');os.replace(folder/'resume.partial.pt',folder/'resume.pt')
                print(args.variant+'_EPOCH_DONE',epoch+1,flush=True)
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(folder/'final.safetensors'))
        if args.smoke:
            changed=[n for n,p in model.named_parameters() if not torch.equal(p.detach().cpu(),before[n])]
            assert changed and any('value' in n for n in changed)
        verify(frozen);verify_baseline()
        complete=dict(status='PASS' if args.smoke else 'COMPLETE',variant=args.variant,updates=updates,
            epochs=0 if args.smoke else cfg['epochs'],samples=len(dataset),checkpoint_sha256=sha(folder/'final.safetensors'),
            elapsed_seconds=elapsed_prior+time.time()-start,baseline_frozen=True,temporal=args.variant=='D',
            history_schedule='Fixed frame modulo 8, three strictly past snapshots',checkpoint_selection='Last fixed epoch')
        save(folder/'COMPLETE.json',complete);save(folder/'PROGRESS.json',complete)
        print(args.variant+'_COMPLETE',folder,flush=True)
if __name__=='__main__':main()
