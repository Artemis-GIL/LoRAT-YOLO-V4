"""B default action, learned candidate values, and separately confirmed recovery."""
import numpy as np
import torch
from safetensors.torch import load_file
from .common import *
from .history import OnlineHistory,baseline_index
from .model_d import ActionValue
from research.bc_r1.tracker_c import ResearchCore,ResearchTracker,PoolController
from tracking.recovery_controller import Mode,association_affinity

class ActionController(PoolController):
    def __init__(self,cfg,core,model):
        self.action_model=model
        super().__init__(cfg,core,comparator=None)
    def reset(self):
        super().reset();self.history=OnlineHistory();self.pending_ids=set();self.pending_box=None;self.pending_frame=None;self.pending_streak=0
    def clear_pending(self):
        self.pending_ids=set();self.pending_box=None;self.pending_frame=None;self.pending_streak=0
    def decide(self,image,local,detections,frame,probe):
        result=super().decide(image,local,detections,frame,probe)
        groups=self.last_groups;features=self.last_features;metadata=self.last_metadata
        default=baseline_index(groups,result['bbox']);selected=default
        result['b_default_action']={k:result.get(k) for k in ['bbox','proposed_commit','commit_source','state','reason','output_source']}
        result['d_action']='KEEP_B';result['d_history_clock_stride']=8
        if default>=0:
            values=self.history.query(groups,features,metadata,groups[default]['box'],frame)
            tensors={k:torch.from_numpy(v)[None].cuda() for k,v in values.items()}
            with torch.inference_mode():
                prediction=self.action_model(**tensors,baseline_index=torch.tensor([default],device='cuda'))
            quality=prediction['quality'][0].cpu().tolist();advantage=prediction['advantage'][0].cpu().tolist()
            candidate=int(np.argmax(advantage))
            # Zero is the predicted gain over the B action, not a tuned identity threshold.
            if advantage[candidate]>0:selected=candidate
            result['comparison']=dict(quality=quality,advantage=advantage,selected=selected,unmatched_index=len(groups),baseline_index=default)
            result['d_history_observations']=[int(v) for v in values['history_valid'][:,:-1].sum(1)]
        else:result['d_action']='KEEP_B_UNREPRESENTABLE_DEFAULT'
        self.history.observe(groups,features,metadata,frame)
        if selected<0 or selected==default:
            self.clear_pending();return result
        chosen=groups[selected];box=chosen['box'];ids=set(chosen['member_ids'])-{0}
        # A learned local preference still carries no new globally observed identity.
        # It may become an alternate action, but cannot lend local streaks to global recovery.
        token_ids={0} if 0 in chosen['member_ids'] else ids
        connected=(self.pending_frame==frame-1 and bool(token_ids&self.pending_ids) and self.pending_box is not None
            and association_affinity(self.pending_box,box,self.cfg) is not None)
        self.pending_streak=self.pending_streak+1 if connected else 1
        self.pending_ids=(token_ids&self.pending_ids) if connected else token_ids
        self.pending_box=list(box);self.pending_frame=frame
        result['d_candidate_streak']=self.pending_streak
        if self.pending_streak<self.cfg.confirmation_frames:
            result['d_action']='KEEP_B_WHILE_ALTERNATIVE_CONFIRMS'
            return result
        source='local' if 0 in chosen['member_ids'] else 'global_verification'
        self.mode=Mode.TRACKING
        result.update(bbox=list(box),proposed_commit=list(box),commit_source=source,output_source=source,state=self.mode.value,
            reason='RECOVERY_CONFIRMED',localization_status='confirmed',output_confirmed=True,d_action='REPLACE_B_CONFIRMED')
        return result

def build(variant,audit,checkpoint=None):
    from tracking.online_tracker import CurrentFrameDetector
    assert variant in ('B','D_ZERO','D0','D','E_ZERO','E')
    assert (checkpoint is not None)==(variant in ('D0','D','E')),'Trained and zero-control variants must use distinct checkpoint semantics'
    setup(20261004);is_d=variant.startswith('D')
    core=ResearchCore(audit,B_WEIGHTS,features=is_d)
    model=None
    if is_d:
        model=ActionValue(temporal=variant!='D0').cuda().eval()
        if checkpoint:model.load_state_dict(load_file(str(checkpoint)),strict=True)
    if variant.startswith('E'):
        from .spatial_e import install_spatial
        install_spatial(core.native.adapter,checkpoint)
    try:detector=CurrentFrameDetector()
    except BaseException:
        del core;torch.cuda.empty_cache();raise
    tracker=ResearchTracker(core,detector)
    if is_d:tracker.controller=ActionController(tracker.config,core,model)
    return tracker
