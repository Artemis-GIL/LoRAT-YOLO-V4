"""Train only the C set decoder on stored, causally generated B features."""
import argparse,fcntl,time,os
from collections import OrderedDict,Counter
import numpy as np
import torch
from safetensors.torch import save_file
from .common import *
from .model_c import CandidateComparator,comparison_loss

class Samples:
    def __init__(self,root):
        self.index=[];self.cache=OrderedDict();self.cache_bytes=0;self.categories={k:[] for k in range(4)}
        for path in sorted(Path(root).glob('UAV*/samples.npz')):
            with np.load(path) as d:
                for i,cat in enumerate(d['category']):
                    self.categories[int(cat)].append(len(self.index));self.index.append((path,i))
        assert self.index
    def get(self,index,rng):
        path,i=self.index[index]
        if path not in self.cache:
            with np.load(path) as f:self.cache[path]={k:f[k] for k in f.files}
            self.cache_bytes+=sum(x.nbytes for x in self.cache[path].values())
            while self.cache_bytes>24*2**30 and len(self.cache)>1:
                _,old=self.cache.popitem(last=False);self.cache_bytes-=sum(x.nbytes for x in old.values())
        self.cache.move_to_end(path);d=self.cache[path];a,b=d['pointers'][i:i+2];n=b-a;order=rng.permutation(n)
        # Each prior sample adds one unmatched target entry beyond its candidates.
        y=d['target'][a+i:b+i+1]
        return d['features'][a:b][order],d['metadata'][a:b][order],np.r_[y[:-1][order],y[-1]],d['iou'][a:b][order],d['selection_valid'][i]

def make_batch(samples):
    n=len(samples);k=max(len(s[0]) for s in samples)
    f=torch.zeros(n,k,4,1024);m=torch.zeros(n,k,4);target=torch.zeros(n,k+1);quality=torch.zeros(n,k)
    padding=torch.ones(n,k,dtype=torch.bool);valid=torch.zeros(n);mask=torch.zeros(n,k)
    for i,(features,meta,y,q,selection_valid) in enumerate(samples):
        count=len(features);f[i,:count]=torch.from_numpy(features.astype(np.float32));m[i,:count]=torch.from_numpy(meta)
        target[i,:count]=torch.from_numpy(y[:-1]);target[i,-1]=float(y[-1]);quality[i,:count]=torch.from_numpy(q)
        padding[i,:count]=False;mask[i,:count]=1.;valid[i]=float(selection_valid)
    return [x.cuda() for x in (f,m,target,quality,padding,mask,valid)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    setup();cfg=read(RUN/'PROTOCOL.json')['C'];folder=RUN/('C_smoke' if args.smoke else 'C');folder.mkdir(exist_ok=True)
    root=RUN/('C_mining_smoke' if args.smoke else 'C_mining');assert (root/'COMPLETE.json').exists()
    assert sha(root/'SUMMARY.json')==read(root/'COMPLETE.json')['summary_sha256']
    verify(read(root/'SUMMARY.json')['sample_files'])
    verify(read(root/'FREEZE.json'))
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=[Path(__file__),Path(__file__).with_name('model_c.py'),Path(__file__).with_name('common.py'),RUN/'PROTOCOL.json',root/'SUMMARY.json']
        files+=list(root.glob('UAV*/samples.npz'))
        frozen=freeze_stage(folder.name,files)
        if (folder/'COMPLETE.json').exists():verify(frozen);return
        ds=Samples(root);model=CandidateComparator().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
        pools=[v for v in ds.categories.values() if v];rng=np.random.default_rng(20261003)
        # Fixed balanced strata sampling; no validation-driven class reweighting.
        steps_per_epoch=max(1,min(4096,(len(ds.index)+15)//16));start=time.time();updates=0;start_epoch=0
        if (folder/'resume.pt').exists():
            state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            rng.bit_generator.state=state['rng'];updates=state['updates'];start_epoch=state['next_epoch']
        with (folder/'train.jsonl').open('a',buffering=1) as trace:
            for epoch in range(start_epoch,1 if args.smoke else cfg['epochs']):
                losses=[]
                for step in range(min(2,steps_per_epoch) if args.smoke else steps_per_epoch):
                    ids=[int(rng.choice(pools[int(rng.integers(len(pools)))])) for _ in range(16)]
                    batch=make_batch([ds.get(i,rng) for i in ids]);f,m,y,q,padding,mask,valid=batch
                    logits,quality=model(f,m,padding);loss,components=comparison_loss(logits,quality,y,q,mask,valid)
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    assert torch.isfinite(grad);optimizer.step();updates+=1;losses.append(float(loss.detach()))
                    row=dict(epoch=epoch+1,step=step+1,updates=updates,loss=float(loss.detach()),**{k:float(v) for k,v in components.items()})
                    trace.write(json.dumps(row)+'\n')
                    if updates%100==0:save(folder/'PROGRESS.json',row);print('C_TRAIN',json.dumps(row),flush=True)
                save(folder/f'epoch_{epoch+1:02d}.json',dict(epoch=epoch+1,mean_loss=float(np.mean(losses)),updates=updates))
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(folder/f'epoch_{epoch+1:02d}.safetensors'))
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                    rng=rng.bit_generator.state,next_epoch=epoch+1,updates=updates),folder/'resume.partial.pt')
                os.replace(folder/'resume.partial.pt',folder/'resume.pt')
                print('C_EPOCH_DONE',epoch+1,flush=True)
        final=folder/'final.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(final))
        verify(frozen);save(folder/'COMPLETE.json',dict(status='SMOKE_PASS' if args.smoke else 'COMPLETE',updates=updates,
            samples=len(ds.index),categories={k:len(v) for k,v in ds.categories.items()},checkpoint_sha256=sha(final),elapsed_seconds=time.time()-start,
            training_scope='comparison head only; B LoRAT unchanged',checkpoint_selection='last fixed epoch'))
        print('C_COMPLETE',folder,flush=True)
if __name__=='__main__':main()
