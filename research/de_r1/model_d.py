"""Matched static/temporal action-value heads relative to the explicit B action."""
import torch
from torch import nn
from torch.nn import functional as F

class ActionValue(nn.Module):
    def __init__(self,temporal=True,dim=128):
        super().__init__();self.temporal=temporal
        self.project=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,64),nn.GELU())
        self.embed=nn.Sequential(nn.Linear(4*64+4+5,dim),nn.LayerNorm(dim),nn.GELU())
        self.cell=nn.GRUCell(dim,dim)
        self.value=nn.Sequential(nn.Linear(dim*3,dim),nn.GELU(),nn.Linear(dim,1))
        # A fresh head takes KEEP_B exactly, before it has learned evidence.
        nn.init.zeros_(self.value[-1].weight);nn.init.zeros_(self.value[-1].bias)
    def forward(self,features,metadata,geometry,history_valid,baseline_index,padding=None):
        n,k,t=features.shape[:3]
        embedded=self.embed(torch.cat((self.project(features).flatten(-2),metadata,geometry),-1))
        state=embedded.new_zeros((n*k,embedded.shape[-1]))
        for step in range(t):
            valid=history_valid[:,:,step]
            if not self.temporal and step<t-1:valid=torch.zeros_like(valid)
            proposal=self.cell(embedded[:,:,step].reshape(n*k,-1),state)
            state=torch.where(valid.reshape(-1,1),proposal,state)
        state=state.reshape(n,k,-1)
        default=state[torch.arange(n,device=state.device),baseline_index][:,None].expand(-1,k,-1)
        logits=self.value(torch.cat((state,default,state-default),-1)).squeeze(-1)
        quality=logits.sigmoid();base=quality.gather(1,baseline_index[:,None])
        advantage=quality-base
        if padding is not None:advantage=advantage.masked_fill(padding,float('-inf'))
        return dict(logits=logits,quality=quality,advantage=advantage)

def action_loss(output,target_iou,target_delta,padding,baseline_index):
    valid=~padding
    # Every retained sample has an exact representable B action. No class-prior balancing.
    q=F.binary_cross_entropy_with_logits(output['logits'],target_iou,reduction='none')
    quality=(q*valid).sum()/valid.sum().clamp_min(1)
    alternatives=valid.clone();alternatives.scatter_(1,baseline_index[:,None],False)
    delta=output['quality']-output['quality'].gather(1,baseline_index[:,None])
    relative=F.smooth_l1_loss(delta,target_delta,reduction='none')
    relative=(relative*alternatives).sum()/alternatives.sum().clamp_min(1)
    return quality+relative,dict(quality=quality.detach(),relative=relative.detach())
