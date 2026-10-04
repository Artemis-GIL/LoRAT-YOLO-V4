"""B control and C learned selection use the same current-frame proposal pool."""
from types import MethodType
import numpy as np
import torch
from safetensors.torch import load_file
from .common import *
from .model_c import CandidateComparator,forward_features,describe
from tracking.causal_tracker_v3 import TransactionalLoRAT
from tracking.structural_ablations import OutputOnlyControllerV3,OutputOnlyTrackerV3
from tracking.recovery_controller import Config,Mode,iou,spatial_groups,association_affinity
from tracking.reliability_v4 import usable

def key(proposal):return tuple(round(float(x),8) for x in proposal['box']+proposal['roi'])

@torch.inference_mode()
def feature_track(adapter,image):
    params=adapter.provider.get(np.array([378,378]))
    x,_,params=adapter.crop(adapter.tensor(image),np.array([378,378]),params,'bilinear',False,adapter.image_mean)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output,zf,xf=forward_features(adapter.model,adapter.z,adapter.normalize(x),adapter.mask)
    out=adapter.post(output)
    descriptor=describe(zf,xf,adapter.mask,out['box']).squeeze(0).cpu().numpy().astype(np.float16)
    pred=adapter.map_boxes(out['box'].double().cpu().numpy(),adapter.reverse(params))[0]
    adapter.clip(pred,np.array([image.shape[1],image.shape[0]]))
    score=float(out['confidence'][0]);adapter.provider.update(score,pred,np.array([image.shape[1],image.shape[0]]))
    roi=adapter.map_boxes(np.array([0.,0.,378.,378.]),adapter.reverse(params));roi[2:]-=roi[:2];pred[2:]-=pred[:2]
    box=pred.tolist();roi=roi.tolist()
    adapter._descriptors[key(dict(box=box,roi=roi))]=descriptor
    return box,dict(roi=roi,score=score,flag='LOCAL',search_pixels=378,reference_updated=False)

class ResearchCore(TransactionalLoRAT):
    def __init__(self,audit,weights=None,features=False):
        super().__init__(audit)
        if weights:
            state=load_file(str(weights));model=self.native.adapter.model
            names={n for n,p in model.named_parameters() if p.requires_grad};assert set(state)==names
            result=model.load_state_dict(state,strict=False)
            assert not result.unexpected_keys and not(names&set(result.missing_keys))
        self.features=features
        if features:
            self.native.adapter.track=MethodType(feature_track,self.native.adapter)
            self.native.adapter._descriptors={}
    def begin_frame(self):
        if self.features:self.native.adapter._descriptors={}
    def descriptor(self,proposal):return self.native.adapter._descriptors[key(proposal)]

class PoolController(OutputOnlyControllerV3):
    def __init__(self,cfg,core,comparator=None):
        self.core=core;self.comparator=comparator
        super().__init__(cfg)
    def reset(self):
        super().reset();self.selected_box=None;self.selected_streak=0
        self.last_features=None;self.last_metadata=None;self.last_groups=[]
    def decide(self,image,local,detections,frame,probe):
        result=super().decide(image,local,detections,frame,probe)
        items=[];extra=0
        if usable(local):items.append(dict(id=0,box=local['box'],proposal=local,detector_score=0.))
        logs={b['id']:b for b in result['verification_branches']}
        for b in self.bank.branches:
            if b.id in logs:proposal=logs[b.id]['proposal']
            elif usable(local) and iou(local['box'],b.detection['box'])>=self.cfg.same_object_iou:proposal=local
            else:
                anchor=b.shadow_box if b.shadow_box is not None else b.detection['box']
                proposal=probe(image,anchor);extra+=1
            if usable(proposal):items.append(dict(id=b.id,box=proposal['box'],proposal=proposal,detector_score=b.detection['score']))
        groups=spatial_groups(items,self.cfg)
        representatives=[min(g,key=lambda x:x['id']) for g in groups]
        features=[];metadata=[];public=[]
        for group,rep in zip(groups,representatives):
            box=rep['box'];features.append(self.core.descriptor(rep['proposal']))
            metadata.append([rep['proposal']['score'],box[2]/image.shape[1],box[3]/image.shape[0],max(e['detector_score'] for e in group)])
            public.append(dict(box=list(box),member_ids=[e['id'] for e in group],score=rep['proposal']['score']))
        self.last_features=np.stack(features) if features else np.zeros((0,4,1024),dtype=np.float16)
        self.last_metadata=np.asarray(metadata,dtype=np.float32).reshape(-1,4);self.last_groups=public
        result.update(comparison_pool=public,extra_pool_probes=extra,probe_count=result['probe_count']+extra)
        if self.comparator is None:return result
        with torch.inference_mode():
            f=torch.from_numpy(self.last_features).float().cuda()[None]
            m=torch.from_numpy(self.last_metadata).cuda()[None]
            logits,quality=self.comparator(f,m)
            probabilities=logits.softmax(-1)[0].cpu().tolist();selection=int(logits[0].argmax())
            quality=quality.sigmoid()[0].cpu().tolist()
        result['comparison']=dict(probabilities=probabilities,quality=quality,selected=selection,unmatched_index=len(public))
        commit=None;source=None
        if selection==len(public):
            self.selected_box=None;self.selected_streak=0;self.mode=Mode.SEARCHING
            output=local['box'] if usable(local) else None
            reason='COMPARISON_UNMATCHED';output_source='local_tentative' if output is not None else None
        else:
            chosen=public[selection];box=chosen['box'];local_selected=0 in chosen['member_ids']
            connected=self.selected_box is not None and association_affinity(self.selected_box,box,self.cfg) is not None
            self.selected_streak=self.selected_streak+1 if connected else 1;self.selected_box=list(box)
            if (result['previous_state']=='TRACKING' and local_selected) or self.selected_streak>=self.cfg.confirmation_frames:
                commit=list(box);source='local' if local_selected else 'global_verification';self.mode=Mode.TRACKING
                reason='LOCAL_COMPARISON_CONFIRMED' if result['previous_state']=='TRACKING' and local_selected else 'RECOVERY_CONFIRMED'
            else:self.mode=Mode.VERIFYING;reason='COMPARISON_WAITING_FOR_CONFIRMATION'
            output=list(box);output_source=source if commit is not None else 'comparison_tentative'
        result.update(bbox=output,proposed_commit=commit,commit_source=source,output_source=output_source,
            state=self.mode.value,reason=reason,localization_status='confirmed' if commit is not None else ('tentative' if output is not None else 'unavailable'),
            output_confirmed=commit is not None,comparison_confirmation_streak=self.selected_streak)
        return result

class ResearchTracker(OutputOnlyTrackerV3):
    def __init__(self,core,detector,cfg=None,pool=False,comparator=None):
        super().__init__(core,detector,cfg)
        if pool:self.controller=PoolController(self.config,core,comparator)
    def track(self,rgb):
        self.core.begin_frame();return super().track(rgb)

def build(variant,audit,checkpoint_b=None,checkpoint_c=None):
    from tracking.online_tracker import CurrentFrameDetector
    assert variant in ('A','B','B_POOL','C')
    setup();core=ResearchCore(audit,None if variant=='A' else checkpoint_b,features=variant in ('B_POOL','C'))
    comparator=None
    if variant=='C':
        comparator=CandidateComparator().cuda().eval()
        comparator.load_state_dict(load_file(str(checkpoint_c)),strict=True)
    try:detector=CurrentFrameDetector()
    except BaseException:
        del core;torch.cuda.empty_cache();raise
    return ResearchTracker(core,detector,pool=variant in ('B_POOL','C'),comparator=comparator)
