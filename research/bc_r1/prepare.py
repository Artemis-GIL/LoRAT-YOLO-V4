"""Freeze sequence-level partitions and the first B/C research budget."""
import csv
from collections import Counter
import numpy as np
from .common import *

def main():
    splits={}
    for r in csv.DictReader(MANIFEST.open()):
        assert r['split'] in ('train','val')
        assert r['sequence'] not in splits or splits[r['sequence']]==r['split']
        splits[r['sequence']]=r['split']
    cases=[];counts=Counter();inputs={str(MANIFEST):sha(MANIFEST)}
    for name,split in sorted(splits.items()):
        folder=DATA/name;g=np.loadtxt(folder/'groundtruth_rect.txt',delimiter=',').reshape(-1,4)
        absent=np.loadtxt(folder/'absent.txt').reshape(-1);videos=list(folder.glob('*.mp4'))
        assert len(videos)==1 and len(g)==len(absent) and np.isfinite(g).all()
        assert absent[0]==0 and (g[0,2:]>0).all()
        cases.append(dict(sequence=name,split=split,video=str(videos[0]),frames=len(g),initial_bbox=g[0].tolist(),
            video_size=videos[0].stat().st_size,video_mtime_ns=videos[0].stat().st_mtime_ns))
        for p in [folder/'groundtruth_rect.txt',folder/'absent.txt']:inputs[str(p)]=sha(p)
        counts[split]+=1
    assert counts==dict(train=1268,val=132)
    protocol=dict(version='bc_r1',seed=20261003,base='V3 output separation; no reverse verification',
        cases=cases,B=dict(epochs=8,samples_per_epoch=4096,microbatch=2,accumulation=8,
            learning_rate=2e-5,minimum_learning_rate=2e-6,weight_decay=.1,warmup_steps=64,
            workers=4,classification='official IoU-aware BCE',regression='official GIoU',
            sample='uniform sequence; 50% uniformly visible frame, 50% small half within sequence; fixed first-frame template',
            checkpoint_selection='last epoch fixed before training; no Test selection',
            search_factor=5.,template_factor=2.,scale_jitter=.25,translation_jitter=3.,
            trainable='existing LoRA, token type embeddings and MLP head; original DINO backbone frozen'),
        C=dict(backbone='B final checkpoint, frozen during first comparison-head experiment',
            feature='shared fused template foreground/background, candidate ROI and search context; no second image encoder',
            pool='local plus all current active YOLO candidates; deduplicate spatially',
            mining='all 1268 training sequences causally; no later GT passed to tracker; labels only in sample writer',
            storage='retain every 8th frame plus wrong-output and multiple-spatial-candidate training frames',
            epochs=12,learning_rate=1e-4,dimension=128,layers=2,heads=4,
            supervision='setwise target selection including unmatched; separate localization quality',
            confirmation_frames=3),
        validation='All 132 complete YOLO-held-out validation sequences; development validation, not an unseen blind test',
        evaluations=['A original output baseline','B fine-tuned output baseline','B_POOL same proposals without learned selection','C learned selection'],
        no_test_inference=True,no_threshold_sweep=True,no_physical_absence_classifier=True,
        limitations=['Single fixed seed initial research run; no claim of convergence or stable gain.',
                     'Existing sequence partition; common source/capture grouping beyond sequence IDs is not available.',
                     'Validation split was already used for YOLO checkpoint selection.'])
    for filename,value in [('PROTOCOL.json',protocol),('INPUT_SHA256.json',inputs)]:
        p=RUN/filename
        if p.exists():assert read(p)==value
        else:save(p,value)
    save(RUN/'STATUS.json',dict(stage='PREPARED',train_sequences=1268,val_sequences=132,
        training_started=False,comparison_training_started=False))
    print('BC_PREPARED',1268,132,flush=True)
if __name__=='__main__':main()
