"""Deterministic supervised Train pairs using the native LoRAT crop geometry."""
from collections import OrderedDict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import DATA, LORAT
import importlib.util
from trackit.core.utils.siamfc_cropping import (get_siamfc_cropping_params, prepare_siamfc_cropping_with_augmentation,
    apply_siamfc_cropping, apply_siamfc_cropping_to_boxes)
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
_spec=importlib.util.spec_from_file_location('bc_official_labels',LORAT/'trackit/data/methods/siamese_tracker_train/transform/default/plugin/box_with_score_map_label_gen/__init__.py')
_labels=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_labels)
positive_sample_assignment=_labels.positive_sample_assignment

def xyxy(box):
    b=np.array(box,dtype=np.float64).copy();b[2:]+=b[:2];return b
def norm(image):
    return (image/255.-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
def image_tensor(rgb): return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float()

class PairDataset(Dataset):
    def __init__(self,cases,config,epoch,seed):
        self.cases=cases;self.cfg=config;self.epoch=epoch;self.seed=seed;self.labels=[]
        for c in cases:
            g=np.loadtxt(DATA/c['sequence']/'groundtruth_rect.txt',delimiter=',').reshape(-1,4)
            a=np.loadtxt(DATA/c['sequence']/'absent.txt').reshape(-1)
            visible=np.flatnonzero((a==0)&(g[:,2:]>0).all(1));visible=visible[visible>0]
            assert len(visible)>0,c['sequence']
            small=visible[np.argsort(g[visible,2]*g[visible,3])[:max(1,len(visible)//2)]]
            self.labels.append((g,visible,small))
        self.caps=OrderedDict();self.first=OrderedDict()
    def __len__(self):return self.cfg['samples_per_epoch']
    def _frame(self,case,index):
        video=case['video']
        if index==0 and video in self.first:return self.first[video].copy()
        if video not in self.caps:
            self.caps[video]=cv2.VideoCapture(video)
            if len(self.caps)>4:self.caps.popitem(last=False)[1].release()
        cap=self.caps[video];self.caps.move_to_end(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read()
        if not ok:raise RuntimeError(f'Cannot decode {video}:{index+1}')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        if index==0:
            self.first[video]=rgb
            if len(self.first)>16:self.first.popitem(last=False)
        return rgb.copy()
    def __getitem__(self,index):
        cv2.setNumThreads(0);torch.set_num_threads(1)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
        ci=int(rng.integers(len(self.cases)));case=self.cases[ci];g,visible,small=self.labels[ci]
        frame=int(rng.choice(small if rng.random()<.5 else visible))
        zrgb=self._frame(case,0);xrgb=self._frame(case,frame);zb=xyxy(g[0]);xb=xyxy(g[frame])
        if rng.random()<.5:
            zrgb=zrgb[:,::-1].copy();xrgb=xrgb[:,::-1].copy()
            zb[[0,2]]=zrgb.shape[1]-zb[[2,0]];xb[[0,2]]=xrgb.shape[1]-xb[[2,0]]
        zp=get_siamfc_cropping_params(zb,2.,np.array([196,196]))
        z,_,zp=apply_siamfc_cropping(image_tensor(zrgb),np.array([196,196]),zp,'bilinear',False)
        zcrop=apply_siamfc_cropping_to_boxes(zb,zp)
        mb=bbox_rasterize(zcrop/14,dtype=np.int64);mb=np.clip(mb,0,14)
        mask=torch.zeros((14,14),dtype=torch.long);mask[mb[1]:mb[3],mb[0]:mb[2]]=1
        assert mask.any(),case['sequence']
        for _ in range(32):
            xp,positive=prepare_siamfc_cropping_with_augmentation(xb,5.,np.array([378,378]),.25,3.,rng,
                min_object_size=np.array([10.,10.]))
            if not positive:continue
            x,_,xp=apply_siamfc_cropping(image_tensor(xrgb),np.array([378,378]),xp,'bilinear',False)
            target=np.clip(apply_siamfc_cropping_to_boxes(xb,xp),0,378)
            if (target[2:]-target[:2] <= 0).any():continue
            indices=positive_sample_assignment(target,np.array([27,27]),np.array([378,378]))
            if len(indices):break
        else:raise RuntimeError(f'Cannot form a valid supervised crop: {case["sequence"]}:{frame+1}')
        # Mild photometric variation; no geometry changes after label generation.
        for crop in (z,x):
            gain=float(rng.uniform(.8,1.2));crop.mul_(gain).clamp_(0,255)
        return dict(z=norm(z),x=norm(x),mask=mask,box=torch.tensor(target/378,dtype=torch.float32),
            indices=torch.from_numpy(indices),sequence=case['sequence'],frame=frame+1)

def collate(batch):
    bi=torch.cat([torch.full((len(x['indices']),),i,dtype=torch.long) for i,x in enumerate(batch)])
    pi=torch.cat([x['indices'] for x in batch])
    return dict(z=torch.stack([x['z'] for x in batch]),x=torch.stack([x['x'] for x in batch]),
        mask=torch.stack([x['mask'] for x in batch]),
        target=dict(boxes=torch.stack([x['box'] for x in batch]),positive_sample_batch_dim_indices=bi,
                    positive_sample_map_dim_indices=pi,num_positive_samples=torch.tensor([len(pi)],dtype=torch.float32)),
        samples=[dict(sequence=x['sequence'],frame=x['frame']) for x in batch])
