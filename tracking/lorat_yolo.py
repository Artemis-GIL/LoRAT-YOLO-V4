"""LoRAT local core and read-only YOLO proposal probing.

Uses the R1-validated implementation unchanged. Proposal scores do not authorize
an automatic state commit; callers must explicitly commit an accepted box.
"""
from pathlib import Path
import importlib.util
import numpy as np
import torch

DEFAULT_ADAPTER = Path('/home/xu/zyx/send/LOCAL_TRACKER_SELECTION_R1_20261002/scripts/adapters.py')

def _box(box):
    b=np.asarray(box,dtype=np.float64)
    if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=0).any():
        raise ValueError('Expected finite xywh with positive width and height')
    return b.copy()

class LoRATCore:
    def __init__(self,audit_path,adapter_path=DEFAULT_ADAPTER):
        spec=importlib.util.spec_from_file_location('_uav_frozen_lorat_adapter',adapter_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.adapter=module.LoRATLocal(audit_path)
        self.initialized=False

    @torch.inference_mode()
    def initialize(self,image_rgb,initial_box):
        self.adapter.initialize(image_rgb,_box(initial_box),'')
        self.initialized=True

    @torch.inference_mode()
    def track(self,image_rgb):
        if not self.initialized:raise RuntimeError('Initialize from the first frame first')
        box,trace=self.adapter.track(image_rgb)
        return dict(box=box,score=trace['score'],roi=trace['roi'])

    @torch.inference_mode()
    def probe_candidates(self,image_rgb,candidates):
        """candidates: [{'box': xywh, 'score': detector_score}, ...].

        Returns records in detector order. Each search uses the fixed initial
        template and an independent candidate anchor. Local state is restored
        even on failure. No language, GT or visibility labels are accepted.
        """
        if not self.initialized:raise RuntimeError('Initialize from the first frame first')
        validated=[(_box(c['box']),float(c['score'])) for c in candidates]
        if not all(np.isfinite(s) for _,s in validated):raise ValueError('Nonfinite detector score')
        original=self.adapter.provider;results=[]
        try:
            for index,(box,detector_score) in enumerate(validated):
                anchor=box.copy();anchor[2:]+=anchor[:2]
                provider=self.adapter.Provider(5.,10);provider.initialize(anchor)
                self.adapter.provider=provider
                refined,trace=self.adapter.track(image_rgb)
                valid=bool(np.isfinite(refined).all() and min(refined[2:])>0 and np.isfinite(trace['score']))
                results.append(dict(index=index,detector_box=box.tolist(),detector_score=detector_score,
                    refined_box=refined,target_score=trace['score'],valid=valid,roi=trace['roi']))
        finally:self.adapter.provider=original
        return results

    @staticmethod
    def rank(probes):
        # Stable detector-order tie breaking; no score fusion or tuned threshold.
        return sorted([p for p in probes if p['valid']],key=lambda p:(-p['target_score'],p['index']))

    def commit(self,box,image_shape):
        """Explicitly move search anchor; retain the original target template."""
        if not self.initialized:raise RuntimeError('Initialize from the first frame first')
        b=_box(box);b[2:]+=b[:2]
        self.adapter.clip(b,np.array([image_shape[1],image_shape[0]]))
        if (b[2:]<=b[:2]).any():raise ValueError('Accepted box lies outside the image')
        self.adapter.provider.initialize(b)

class LoRATYOLOBridge:
    def __init__(self,core,detector=None,topk=12):
        self.core=core;self.detector=detector;self.topk=int(topk)
        if self.topk<=0:raise ValueError('topk must be positive')

    def from_candidates(self,image_rgb,candidates):
        ordered=sorted(candidates,key=lambda c:-c['score'])[:self.topk]
        return self.core.rank(self.core.probe_candidates(image_rgb,ordered))

    def propose(self,image_rgb):
        if self.detector is None:raise RuntimeError('A YOLO detector with infer(BGR) is required')
        bgr=np.ascontiguousarray(image_rgb[:,:,::-1])
        return self.from_candidates(image_rgb,self.detector.infer(bgr))
