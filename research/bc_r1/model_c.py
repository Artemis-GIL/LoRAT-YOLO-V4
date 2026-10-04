"""Shared LoRAT features and a permutation-equivariant candidate-set decoder."""
import torch
from torch import nn
from torch.nn import functional as F

def forward_features(model,z,x,mask):
    zf=model._z_feat(z,mask);xf=model._x_feat(x)
    tokens=torch.cat((zf,xf),1)
    for block in model.blocks:tokens=block(tokens)
    tokens=model.norm(tokens);zf=tokens[:,:zf.shape[1]];xf=tokens[:,zf.shape[1]:]
    return model.head(xf),zf,xf

def describe(zf,xf,mask,crop_box):
    """Pool first-template FG/BG, current predicted ROI, and search context."""
    weights=mask.flatten(1).float()
    fg=(zf.float()*weights[:,:,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
    bg=(zf.float()*(1-weights[:,:,None])).sum(1)/(1-weights).sum(1,keepdim=True).clamp_min(1)
    b=crop_box.detach().float().clamp(0,378)/378
    t=torch.tensor([1/6,1/2,5/6],device=xf.device)
    xx=b[:,0,None]+(b[:,2]-b[:,0])[:,None]*t
    yy=b[:,1,None]+(b[:,3]-b[:,1])[:,None]*t
    grid=torch.stack((xx[:,None,:].expand(-1,3,-1),yy[:,:,None].expand(-1,-1,3)),-1)*2-1
    image=xf.float().transpose(1,2).reshape(-1,xf.shape[-1],27,27)
    roi=F.grid_sample(image,grid,mode='bilinear',padding_mode='border',align_corners=False).mean((2,3))
    return torch.stack((fg,bg,roi,xf.float().mean(1)),1)

class CandidateComparator(nn.Module):
    def __init__(self,feature_dim=1024,dim=128,layers=2,heads=4):
        super().__init__()
        self.projection=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,dim),nn.GELU())
        self.combine=nn.Sequential(nn.Linear(4*dim+4,dim),nn.GELU(),nn.LayerNorm(dim))
        self.null=nn.Parameter(torch.zeros(1,1,dim))
        self.attention=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,heads,dim*2,dropout=0.,
            activation='gelu',batch_first=True,norm_first=True),layers,enable_nested_tensor=False)
        self.selection=nn.Linear(dim,1);self.quality=nn.Linear(dim,1)
    def forward(self,features,metadata,padding=None):
        # No source ID or positional embedding: candidate order has no meaning.
        h=self.projection(features).flatten(-2);h=self.combine(torch.cat((h,metadata),-1))
        h=torch.cat((h,self.null.expand(h.shape[0],-1,-1)),1)
        if padding is not None:padding=torch.cat((padding,torch.zeros((len(h),1),dtype=torch.bool,device=h.device)),1)
        h=self.attention(h,src_key_padding_mask=padding)
        logits=self.selection(h).squeeze(-1)
        if padding is not None:logits=logits.masked_fill(padding,float('-inf'))
        return logits,self.quality(h[:,:-1]).squeeze(-1)

def comparison_loss(logits,quality,target,iou,valid,selection_valid):
    logp=F.log_softmax(logits,dim=-1).masked_fill(~torch.isfinite(logits),0.)
    matching=-(target*logp).sum(-1)
    matching=(matching*selection_valid).sum()/selection_valid.sum().clamp_min(1)
    localization=F.binary_cross_entropy_with_logits(quality,iou,reduction='none')
    localization=(localization*valid).sum()/valid.sum().clamp_min(1)
    return matching+localization,dict(selection=matching.detach(),quality=localization.detach())
