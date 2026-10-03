"""Spatial correspondence only; no identity, confidence or visibility labels."""
import numpy as np
def match(local_box,candidates):
    if not candidates:return list(local_box),None,0.
    a=np.asarray(local_box,float);b=np.asarray([c['box'] for c in candidates],float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:2]+a[2:],b[:,:2]+b[:,2:])-np.maximum(a[:2],b[:,:2]));inter=wh.prod(1)
    overlap=inter/np.maximum(np.maximum(a[2:],0).prod()+np.maximum(b[:,2:],0).prod(1)-inter,1e-12)
    idx=int(np.argmax(overlap))
    if overlap[idx]<=0:return a.tolist(),None,0.
    return b[idx].tolist(),idx,float(overlap[idx])
