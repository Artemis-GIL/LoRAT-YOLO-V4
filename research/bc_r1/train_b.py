"""Fine-tune native LoRAT trainable weights with its unchanged official losses."""
import argparse,fcntl,math,time,os
from torch.utils.data import DataLoader
from .common import *
from .data import PairDataset,collate

def criterion():
    from trackit.criteria.methods.box_with_score_map import SimpleCriteria
    from trackit.criteria.modules.iou_loss import GIoULoss
    return SimpleCriteria(torch.nn.BCEWithLogitsLoss(reduction='none'),GIoULoss(),True,1.,1.,'bce','giou').cuda()

def batch_to_cuda(batch):
    for k in ['z','x','mask']:batch[k]=batch[k].cuda(non_blocking=True)
    batch['target']={k:v.cuda(non_blocking=True) for k,v in batch['target'].items()}
    return batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    setup();protocol=read(RUN/'PROTOCOL.json');cfg=protocol['B'];folder=RUN/('B_smoke' if args.smoke else 'B')
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=[Path(__file__),Path(__file__).with_name('common.py'),Path(__file__).with_name('data.py'),RUN/'PROTOCOL.json',RUN/'INPUT_SHA256.json',
            ORIGINAL/'weights/LoRAT_large-378.bin',Path('/home/xu/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth'),ORIGINAL/'scripts/adapters.py']
        source+=list((LORAT/'trackit').rglob('*.py'))
        frozen=freeze_stage(folder.name,source)
        if (folder/'COMPLETE.json').exists():
            verify(frozen);print('B_ALREADY_COMPLETE',flush=True);return
        adapter=load_adapter(folder/'MODEL.json');model=adapter.model;model.train();loss_fn=criterion()
        params=[p for p in model.parameters() if p.requires_grad]
        groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and p.ndim>1 and 'embed' not in n],weight_decay=cfg['weight_decay']),
                dict(params=[p for n,p in model.named_parameters() if p.requires_grad and (p.ndim==1 or 'embed' in n)],weight_decay=0.)]
        opt=torch.optim.AdamW(groups,lr=cfg['learning_rate'])
        start_epoch=0;updates=0
        if (folder/'resume.pt').exists() and not args.smoke:
            state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=False);opt.load_state_dict(state['optimizer'])
            start_epoch=state['next_epoch'];updates=state['updates']
        cases=[x for x in protocol['cases'] if x['split']=='train']
        total=cfg['epochs']*cfg['samples_per_epoch']//(cfg['microbatch']*cfg['accumulation'])
        steps=0;started=time.time();before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad} if args.smoke else None
        frozen_probe=model.patch_embed.proj.weight.detach().cpu().clone() if args.smoke else None
        trace=(folder/'train.jsonl').open('a',buffering=1)
        for epoch in range(start_epoch,1 if args.smoke else cfg['epochs']):
            ds=PairDataset(cases,cfg,epoch,protocol['seed'])
            loader=DataLoader(ds,batch_size=cfg['microbatch'],num_workers=0 if args.smoke else cfg['workers'],
                collate_fn=collate,pin_memory=True,shuffle=False,
                **({} if args.smoke else dict(multiprocessing_context='spawn',prefetch_factor=2)))
            opt.zero_grad(set_to_none=True);running=0.;seen=0
            for index,batch in enumerate(loader):
                batch=batch_to_cuda(batch)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch['z'],batch['x'],batch['mask'])
                result=loss_fn(out,batch['target']);loss=result.loss
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss epoch {epoch} batch {index}')
                accumulation=1 if args.smoke else cfg['accumulation']
                (loss/accumulation).backward();running+=float(loss.detach());seen+=1;steps+=1
                if (index+1)%accumulation==0:
                    grad=torch.nn.utils.clip_grad_norm_(params,1.)
                    assert torch.isfinite(grad)
                    if updates<cfg['warmup_steps']:lr=cfg['learning_rate']*(updates+1)/cfg['warmup_steps']
                    else:
                        frac=(updates-cfg['warmup_steps'])/max(1,total-cfg['warmup_steps'])
                        lr=cfg['minimum_learning_rate']+.5*(cfg['learning_rate']-cfg['minimum_learning_rate'])*(1+math.cos(math.pi*frac))
                    for group in opt.param_groups:group['lr']=lr
                    opt.step();opt.zero_grad(set_to_none=True);updates+=1
                    record=dict(epoch=epoch+1,batch=index+1,updates=updates,loss=float(loss.detach()),lr=lr,
                        grad_norm=float(grad),peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.time()-started)
                    trace.write(json.dumps(record)+'\n')
                    if updates%10==0 or args.smoke:
                        save(folder/'PROGRESS.json',record);print('B_TRAIN',json.dumps(record),flush=True)
                if args.smoke and steps==2:break
            if args.smoke:break
            save_trainable(model,folder/f'epoch_{epoch+1:02d}.safetensors')
            checkpoint=dict(model={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                optimizer=opt.state_dict(),next_epoch=epoch+1,updates=updates)
            torch.save(checkpoint,folder/'resume.partial.pt');os.replace(folder/'resume.partial.pt',folder/'resume.pt')
            save(folder/f'epoch_{epoch+1:02d}.json',dict(epoch=epoch+1,mean_loss=running/seen,updates=updates))
            print('B_EPOCH_DONE',epoch+1,flush=True)
        trace.close();verify(frozen)
        if args.smoke:
            changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(),before[n])]
            assert changed and any('lora' in n for n in changed) and any('head' in n for n in changed)
            assert torch.equal(frozen_probe,model.patch_embed.proj.weight.detach().cpu())
            save(folder/'COMPLETE.json',dict(status='PASS',real_backward_steps=steps,changed_trainable_tensors=len(changed),
                frozen_patch_embedding_unchanged=True,peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30))
        else:
            save_trainable(model,folder/'final.safetensors')
            save(folder/'COMPLETE.json',dict(status='COMPLETE',epochs=cfg['epochs'],updates=updates,
                checkpoint=str(folder/'final.safetensors'),sha256=sha(folder/'final.safetensors'),elapsed_seconds=time.time()-started))
        print('B_COMPLETE',folder,flush=True)
if __name__=='__main__':main()
