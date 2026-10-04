"""Causal B rollouts; supervision is applied only by the training sample writer."""
import argparse,fcntl,time,uuid
from collections import Counter
from types import SimpleNamespace
import numpy as np
import torch
from .common import *
from .tracker_c import build
from tracking.run_tracker_v3 import frame_stream
from tracking.recovery_controller import iou

LABEL_PROTOCOL=dict(version='geometry_association_v1',
    positive='IoU >= 0.5, or center error <= 0.25 GT diagonal with area ratio in [0.5,2]',
    negative='IoU < 0.1 and center error > GT diagonal; all candidates negative on annotated absent frames',
    ambiguity='Other candidates are ambiguous; exclude the entire set from selection loss if any member is ambiguous; quality supervision remains',
    identity_limit='Geometry-derived target association labels, not annotated distractor identities',
    unmatched='No positive or ambiguous member in the current pool; does not assert physical absence')

def labels(boxes,gt,visible):
    k=len(boxes);overlap=np.array([iou(b,gt) if visible else 0. for b in boxes],dtype=np.float32)
    positive=np.zeros(k,dtype=bool);negative=np.ones(k,dtype=bool)
    if visible:
        center=gt[:2]+gt[2:]/2;diagonal=max(float(np.linalg.norm(gt[2:])),1e-6)
        for j,box in enumerate(boxes):
            b=np.asarray(box);distance=float(np.linalg.norm(b[:2]+b[2:]/2-center))
            ratio=float(np.prod(b[2:])/max(np.prod(gt[2:]),1e-6))
            positive[j]=overlap[j]>=.5 or (distance<=.25*diagonal and .5<=ratio<=2.)
            negative[j]=overlap[j]<.1 and distance>diagonal
    valid=bool((positive|negative).all())
    target=np.zeros(k+1,dtype=np.float32)
    if positive.any():target[:k]=positive/positive.sum()
    else:target[k]=1.
    category=0 if not valid else (1 if not positive.any() else (2 if k>1 else 3))
    return target,overlap,valid,category

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    setup();protocol=read(RUN/'PROTOCOL.json');folder=RUN/('C_mining_smoke' if args.smoke else 'C_mining');folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=RUN/'B/final.safetensors'
        assert b.exists() and sha(b)==read(RUN/'B/COMPLETE.json')['sha256']
        sources=[Path(__file__),Path(__file__).with_name('tracker_c.py'),Path(__file__).with_name('model_c.py'),
            Path(__file__).with_name('common.py'),RUN/'PROTOCOL.json',b]
        sources+=list(PROJECT.glob('tracking/*.py'))
        frozen=freeze_stage(folder.name,sources);save(folder/'LABEL_PROTOCOL.json',LABEL_PROTOCOL)
        cases=[c for c in protocol['cases'] if c['split']=='train']
        if args.smoke:cases=cases[:1]
        totals=Counter();summaries=[];started=time.time()
        with build('B_POOL',folder/'MODEL.json',checkpoint_b=b) as tracker:
            for si,case in enumerate(cases,1):
                name=case['sequence'];target=folder/name
                if (target/'COMPLETE.json').exists():
                    done=read(target/'COMPLETE.json');assert sha(target/'samples.npz')==done['samples_sha256']
                    assert sha(target/'trace.jsonl')==done['trace_sha256']
                    summaries.append(done);totals.update(done['counts']);continue
                if target.exists():
                    archive=folder/'interrupted_attempts'/name/uuid.uuid4().hex
                    archive.parent.mkdir(parents=True,exist_ok=True);target.replace(archive)
                target.mkdir()
                # Labels stay in this writer. The runtime receives RGB and the first box only.
                gt=np.loadtxt(DATA/name/'groundtruth_rect.txt',delimiter=',').reshape(-1,4)
                absent=np.loadtxt(DATA/name/'absent.txt').reshape(-1)
                features=[];metadata=[];targets=[];qualities=[];selection=[];category=[];frames=[];pointers=[0]
                counts=Counter();stream=frame_stream(SimpleNamespace(video=Path(case['video']),frames=None,max_frames=8 if args.smoke else None))
                with (target/'trace.jsonl').open('w',buffering=1) as trace:
                    try:
                        for frame,rgb,source in stream:
                            if frame==1:
                                result=tracker.initialize(rgb,case['initial_bbox'])
                                z=tracker.core.native.adapter.z.clone();mask=tracker.core.native.adapter.mask.clone()
                            else:result=tracker.track(rgb)
                            trace.write(json.dumps(result,allow_nan=False)+'\n');counts['frames']+=1
                            if frame==1:continue
                            pool=tracker.controller.last_groups
                            vis=bool(absent[frame-1]==0 and (gt[frame-1,2:]>0).all())
                            output_iou=iou(result['bbox'],gt[frame-1]) if result['bbox'] is not None and vis else 0.
                            keep=bool(frame%8==0 or len(pool)>1 or (vis and output_iou<.5) or (not vis and frame%4==0))
                            if keep and pool:
                                y,q,valid,cat=labels([p['box'] for p in pool],gt[frame-1],vis)
                                features.append(tracker.controller.last_features.copy());metadata.append(tracker.controller.last_metadata.copy())
                                targets.append(y);qualities.append(q);selection.append(valid);category.append(cat);frames.append(frame)
                                pointers.append(pointers[-1]+len(pool));counts['samples']+=1;counts['candidates']+=len(pool);counts[f'category_{cat}']+=1
                            if frame%200==0:
                                progress=dict(sequence_index=si,sequences=len(cases),sequence=name,frame=frame,frames=case['frames'],
                                    completed_frames=totals['frames'],elapsed_seconds=time.time()-started)
                                save(folder/'PROGRESS.json',progress);print('C_MINING',json.dumps(progress),flush=True)
                    finally:stream.close()
                assert counts['frames']==(min(8,case['frames']) if args.smoke else case['frames'])
                assert torch.equal(z,tracker.core.native.adapter.z) and torch.equal(mask,tracker.core.native.adapter.mask)
                # Flat storage with pointers avoids Python pickle and disk image duplication.
                with (target/'samples.partial.npz').open('wb') as f:
                    np.savez_compressed(f,features=np.concatenate(features) if features else np.zeros((0,4,1024),np.float16),
                        metadata=np.concatenate(metadata) if metadata else np.zeros((0,4),np.float32),
                        target=np.concatenate(targets) if targets else np.zeros(0,np.float32),
                        iou=np.concatenate(qualities) if qualities else np.zeros(0,np.float32),
                        pointers=np.asarray(pointers,np.int64),selection_valid=np.asarray(selection,bool),category=np.asarray(category,np.int64),frames=np.asarray(frames,np.int64))
                (target/'samples.partial.npz').replace(target/'samples.npz')
                done=dict(sequence=name,counts=dict(counts),samples_sha256=sha(target/'samples.npz'),trace_sha256=sha(target/'trace.jsonl'),initial_template_preserved=True)
                save(target/'COMPLETE.json',done);summaries.append(done);totals.update(counts)
                print('C_MINING_DONE',si,len(cases),name,dict(counts),flush=True)
        verify(frozen);verify(read(RUN/'INPUT_SHA256.json'))
        save(folder/'SUMMARY.json',dict(sequences=len(summaries),counts=dict(totals),elapsed_seconds=time.time()-started,source_split='train',B_checkpoint_sha256=sha(b),
            sample_files={str(folder/s['sequence']/'samples.npz'):s['samples_sha256'] for s in summaries}))
        save(folder/'COMPLETE.json',dict(summary_sha256=sha(folder/'SUMMARY.json')))
        print('C_MINING_COMPLETE',dict(totals),flush=True)
if __name__=='__main__':main()
