"""D targets and strictly causal fixed-period feature histories from frozen B rollouts.

Stored source features are referenced in place. Preparation never loads a model.
Training query mining may use GT; history writes are exactly frame % 8 == 0,
which is independent of GT and reproduced by the runtime.
"""
import argparse,fcntl,os,time
from collections import Counter,OrderedDict,deque
from pathlib import Path
import numpy as np
import torch
from .common import OLD,RUN,DATA,read,save,sha,verify,freeze_stage

POLICY=dict(version='B_relative_utility_period8_history3_v1',source='frozen C_R1 B_POOL causal train rollouts',
    history_write='Only frame % 8 == 0, after the current sample has read strictly earlier snapshots',
    history_snapshots=3,query_samples='All stored source samples; omit from training only when B output has no exact candidate representation',
    association='Mutual unique member-ID association; ID 0 is a local branch, not target identity',
    supervision='candidate IoU minus B output IoU; annotated absent candidates and B both have utility zero',
    sampling='Uniform among valid stored samples; no former class balancing',
    duplicate_default='Prefer a matching candidate containing local member 0, otherwise the first exact candidate',
    source_features='Referenced directly; no model inference or feature copying to disk')

def overlap(box,gt):
    if box is None:return 0.
    b=np.asarray(box,dtype=np.float64);g=np.asarray(gt,dtype=np.float64)
    inter=np.maximum(np.minimum(b[:2]+b[2:],g[:2]+g[2:])-np.maximum(b[:2],g[:2]),0).prod()
    return float(inter/max(float(b[2:].prod()+g[2:].prod()-inter),1e-15))

def check_source(source):
    done=read(source.parent/'COMPLETE.json')
    assert sha(source)==done['samples_sha256'],source
    assert sha(source.parent/'trace.jsonl')==done['trace_sha256'],source.parent
    return done

def prepare(root=None):
    from .history import history_indices
    root=Path(root or RUN/'D_data');root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        mining=OLD/'C_mining';summary=read(mining/'SUMMARY.json')
        assert sha(mining/'SUMMARY.json')==read(mining/'COMPLETE.json')['summary_sha256']
        verify(read(OLD/'INPUT_SHA256.json'))
        sources=sorted(mining.glob('UAV*/samples.npz'));assert len(sources)==1268
        policy_path=root/'POLICY.json'
        if policy_path.exists():assert read(policy_path)==POLICY
        else:save(policy_path,POLICY)
        files=[Path(__file__),Path(__file__).with_name('history.py'),Path(__file__).with_name('common.py'),
               policy_path,mining/'SUMMARY.json',mining/'COMPLETE.json',OLD/'PROTOCOL.json',OLD/'INPUT_SHA256.json']
        if (RUN/'PROTOCOL.json').exists():files.append(RUN/'PROTOCOL.json')
        frozen=freeze_stage(str(root.relative_to(RUN)),files)
        if (root/'COMPLETE.json').exists():
            done=read(root/'COMPLETE.json');assert sha(root/'SUMMARY.json')==done['summary_sha256']
            verify(read(root/'SUMMARY.json')['label_files']);verify(frozen)
            return read(root/'SUMMARY.json')
        total=Counter();records=[];started=time.time()
        for si,source in enumerate(sources,1):
            name=source.parent.name;target=root/name;target.mkdir(exist_ok=True)
            original=check_source(source);assert original['samples_sha256']==summary['sample_files'][str(source)]
            if (target/'COMPLETE.json').exists():
                done=read(target/'COMPLETE.json');assert sha(target/'labels.npz')==done['labels_sha256']
                assert done['source_samples_sha256']==original['samples_sha256'] and done['source_trace_sha256']==original['trace_sha256']
                records.append(done);total.update(done['counts']);continue
            with np.load(source) as f:
                pointers=f['pointers'];frames=f['frames'];stored_iou=f['iou']
            gt=np.loadtxt(DATA/name/'groundtruth_rect.txt',delimiter=',').reshape(-1,4)
            absent=np.loadtxt(DATA/name/'absent.txt').reshape(-1)
            count=len(stored_iou);n=len(frames);assert len(pointers)==n+1
            boxes=np.zeros((count,4),np.float64);hist=np.full((count,3),-1,np.int64)
            gaps=np.zeros((count,3),np.int64);baseline_index=np.full(n,-1,np.int64)
            baseline_iou=np.zeros(n,np.float32);quality=np.zeros(count,np.float32)
            frame_to_sample={int(frame):j for j,frame in enumerate(frames)}
            snapshots=deque(maxlen=3);counts=Counter(samples=n,candidates=count);seen=0
            with (source.parent/'trace.jsonl').open() as stream:
                for frame,line in enumerate(stream,1):
                    if frame not in frame_to_sample:
                        if frame%8==0:
                            row=__import__('json').loads(line);assert row['frame']==frame
                            assert not row.get('comparison_pool',[]),'Nonempty periodic pool missing from frozen feature samples'
                            snapshots.append(dict(frame=frame,groups=[],row_range=(0,0)))
                            counts['history_snapshots_written']+=1;counts['empty_periodic_snapshots']+=1
                        continue
                    row=__import__('json').loads(line);assert row['frame']==frame;j=frame_to_sample[frame];seen+=1
                    a,b=map(int,pointers[j:j+2]);pool=row['comparison_pool'];assert len(pool)==b-a
                    groups=[dict(box=list(p['box']),member_ids=list(p['member_ids']),row=a+k) for k,p in enumerate(pool)]
                    boxes[a:b]=[g['box'] for g in groups]
                    prior=list(snapshots);indices=np.asarray(history_indices(groups,prior),dtype=np.int64)
                    assert indices.shape==(b-a,3)
                    hist[a:b]=indices
                    padded_times=[None]*(3-len(prior))+[entry['frame'] for entry in prior]
                    for t,past_frame in enumerate(padded_times):
                        mask=indices[:,t]>=0
                        if past_frame is None:assert not mask.any()
                        else:
                            assert past_frame<frame and past_frame%8==0
                            gaps[a:b,t][mask]=frame-past_frame
                            if mask.any():
                                lo,hi=prior[t-(3-len(prior))]['row_range']
                                assert ((indices[:,t][mask]>=lo)&(indices[:,t][mask]<hi)).all()
                    counts['history_slots']+=int((indices>=0).sum())
                    counts['candidates_with_history']+=int((indices>=0).any(1).sum())
                    matches=[k for k,g in enumerate(groups) if row['bbox'] is not None and np.allclose(g['box'],row['bbox'],rtol=0,atol=1e-7)]
                    if matches:
                        local=[k for k in matches if 0 in groups[k]['member_ids']]
                        baseline_index[j]=(local or matches)[0];counts['valid_samples']+=1
                    else:counts['skipped_no_exact_baseline']+=1
                    counts['duplicate_exact_baseline_samples']+=len(matches)>1
                    visible=absent[frame-1]==0 and (gt[frame-1,2:]>0).all()
                    baseline_iou[j]=overlap(row['bbox'],gt[frame-1]) if visible else 0.
                    quality[a:b]=[overlap(g['box'],gt[frame-1]) if visible else 0. for g in groups]
                    assert np.allclose(quality[a:b],stored_iou[a:b],rtol=0,atol=1e-7)
                    if matches:assert abs(float(quality[a+baseline_index[j]]-baseline_iou[j]))<=1e-7
                    # Write only after query extraction. No GT-dependent frame choice.
                    if frame%8==0:
                        snapshots.append(dict(frame=frame,groups=groups,row_range=(a,b)))
                        counts['history_snapshots_written']+=1
            assert seen==n
            repeated_baseline=np.repeat(baseline_iou,np.diff(pointers))
            delta=quality-repeated_baseline
            for j,k in enumerate(baseline_index):
                if k>=0:delta[pointers[j]+k]=0.
            tmp=target/'labels.partial.npz'
            with tmp.open('wb') as f:
                np.savez_compressed(f,pointers=pointers,frames=frames,boxes=boxes,history_indices=hist,history_gaps=gaps,
                    baseline_index=baseline_index,baseline_iou=baseline_iou,quality=quality,target_delta=delta)
            os.replace(tmp,target/'labels.npz')
            done=dict(sequence=name,source=str(source),source_samples_sha256=original['samples_sha256'],source_trace_sha256=original['trace_sha256'],
                labels_sha256=sha(target/'labels.npz'),counts=dict(counts))
            save(target/'COMPLETE.json',done);records.append(done);total.update(counts)
            if si%100==0:
                progress=dict(sequences=si,total_sequences=len(sources),counts=dict(total),elapsed_seconds=time.time()-started)
                save(root/'PROGRESS.json',progress);print('D_DATA',progress,flush=True)
        assert len(records)==1268 and total['samples']==summary['counts']['samples']
        verify(frozen);verify(read(OLD/'INPUT_SHA256.json'))
        result=dict(sequences=len(records),counts=dict(total),elapsed_seconds=time.time()-started,
            label_files={str(root/r['sequence']/'labels.npz'):r['labels_sha256'] for r in records},
            source_samples={r['source']:r['source_samples_sha256'] for r in records},
            source_traces={str(Path(r['source']).parent/'trace.jsonl'):r['source_trace_sha256'] for r in records},policy=POLICY)
        save(root/'SUMMARY.json',result);save(root/'COMPLETE.json',dict(summary_sha256=sha(root/'SUMMARY.json'),status='COMPLETE'))
        print('D_DATA_COMPLETE',dict(total),flush=True);return result

class Samples:
    """Uniform valid-query index with bounded lazy source-feature cache."""
    def __init__(self,root=None,cache_bytes=8*2**30):
        self.root=Path(root or RUN/'D_data');self.index=[];self.cache=OrderedDict();self.cache_bytes=0;self.limit=cache_bytes
        assert (self.root/'COMPLETE.json').exists()
        assert sha(self.root/'SUMMARY.json')==read(self.root/'COMPLETE.json')['summary_sha256']
        self.summary=read(self.root/'SUMMARY.json')
        for path in sorted(self.root.glob('UAV*/labels.npz')):
            assert sha(path)==self.summary['label_files'][str(path)]
            with np.load(path) as d:
                self.index.extend((path,j) for j,k in enumerate(d['baseline_index']) if k>=0)
        assert len(self.index)==self.summary['counts']['valid_samples']
    def __len__(self):return len(self.index)
    def _arrays(self,path):
        if path not in self.cache:
            source=Path(read(path.parent/'COMPLETE.json')['source'])
            assert sha(source)==self.summary['source_samples'][str(source)]
            with np.load(source) as d:original={k:d[k] for k in ('features','metadata')}
            with np.load(path) as d:derived={k:d[k] for k in d.files}
            entry=(original,derived);self.cache[path]=entry
            self.cache_bytes+=sum(a.nbytes for mapping in entry for a in mapping.values())
            while self.cache_bytes>self.limit and len(self.cache)>1:
                _,old=self.cache.popitem(last=False);self.cache_bytes-=sum(a.nbytes for mapping in old for a in mapping.values())
        self.cache.move_to_end(path);return self.cache[path]
    def get(self,index):
        from .history import relative_geometry
        path,j=self.index[int(index)];original,d=self._arrays(path);a,b=map(int,d['pointers'][j:j+2]);k=b-a
        current=np.arange(a,b,dtype=np.int64)[:,None]
        rows=np.concatenate((d['history_indices'][a:b],current),1);valid=rows>=0
        safe=np.maximum(rows,0);features=original['features'][safe].copy();metadata=original['metadata'][safe].copy()
        features[~valid]=0;metadata[~valid]=0
        temporal_boxes=d['boxes'][safe];baseline=int(d['baseline_index'][j]);assert 0<=baseline<k
        base=d['boxes'][a+baseline]
        frame_gaps=np.concatenate((d['history_gaps'][a:b],np.zeros((k,1),np.int64)),1)
        geometry=relative_geometry(temporal_boxes,base,frame_gaps)
        geometry[~valid]=0
        return dict(features=features,metadata=metadata,geometry=geometry,history_valid=valid,
            baseline_index=baseline,target_delta=d['target_delta'][a:b].copy(),quality=d['quality'][a:b].copy(),
            frame=int(d['frames'][j]),sequence=path.parent.name)

def collate(samples):
    """CPU tensors only; the training entry point owns device placement."""
    n=len(samples);assert n;maxk=max(len(s['features']) for s in samples)
    out=dict(features=torch.zeros(n,maxk,4,4,1024),metadata=torch.zeros(n,maxk,4,4),geometry=torch.zeros(n,maxk,4,5),
        history_valid=torch.zeros(n,maxk,4,dtype=torch.bool),padding=torch.ones(n,maxk,dtype=torch.bool),
        baseline_index=torch.zeros(n,dtype=torch.long),target_delta=torch.zeros(n,maxk),quality=torch.zeros(n,maxk))
    for i,s in enumerate(samples):
        k=len(s['features']);out['padding'][i,:k]=False;out['baseline_index'][i]=s['baseline_index']
        for name in ('features','metadata','geometry','history_valid','target_delta','quality'):
            out[name][i,:k]=torch.as_tensor(s[name],dtype=out[name].dtype)
    assert out['history_valid'][:,:,-1].equal(~out['padding'])
    assert torch.all(out['target_delta'][torch.arange(n),out['baseline_index']]==0)
    return out

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();prepare()
