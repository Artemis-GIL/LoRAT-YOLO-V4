"""Fixed-clock causal history. Branch IDs are association cues, never identities."""
import numpy as np

HISTORY=3
STRIDE=8

def links(a,b):
    x=set(a['member_ids']);y=set(b['member_ids'])
    # Local is a path, not a target ID. Its motion and appearance remain inputs.
    return bool((x-{0})&(y-{0})) or (0 in x and 0 in y)

def history_indices(current_groups,past_snapshots):
    snapshots=list(past_snapshots)[-HISTORY:]
    result=np.full((len(current_groups),HISTORY),-1,dtype=np.int64)
    for column,snapshot in enumerate(snapshots,HISTORY-len(snapshots)):
        previous=snapshot['groups']
        matrix=np.array([[links(a,b) for b in previous] for a in current_groups],dtype=bool).reshape(len(current_groups),len(previous))
        for i in range(len(current_groups)):
            js=np.flatnonzero(matrix[i])
            if len(js)==1 and matrix[:,js[0]].sum()==1:
                result[i,column]=previous[int(js[0])]['row']
    return result

def relative_geometry(boxes,baseline_box,gaps):
    boxes=np.asarray(boxes,np.float32);b=np.asarray(baseline_box,np.float32)
    wh=np.maximum(b[2:],1e-6);center=b[:2]+wh/2
    shape=np.maximum(boxes[...,2:],1e-6)
    xy=(boxes[...,:2]+shape/2-center)/wh
    scale=np.log(shape/wh)
    time=np.log1p(np.asarray(gaps,np.float32))/np.float32(np.log(33.))
    return np.concatenate((xy,scale,time[...,None]),axis=-1).astype(np.float32)

def baseline_index(groups,box):
    if box is None:return -1
    found=[i for i,g in enumerate(groups) if np.allclose(g['box'],box,rtol=0,atol=1e-7)]
    if not found:return -1
    return min(found,key=lambda i:(0 not in groups[i]['member_ids'],i))

class OnlineHistory:
    def __init__(self):self.reset()
    def reset(self):self.snapshots=[];self.records={};self.next_row=0;self.last_frame=0
    def query(self,groups,features,metadata,baseline,frame):
        assert frame>self.last_frame
        assert all(s['frame']<frame for s in self.snapshots)
        k=len(groups);f=np.zeros((k,HISTORY+1,4,1024),np.float32);m=np.zeros((k,HISTORY+1,4),np.float32)
        geometry=np.zeros((k,HISTORY+1,5),np.float32);valid=np.zeros((k,HISTORY+1),bool)
        f[:,-1]=features;m[:,-1]=metadata;valid[:,-1]=True
        boxes=np.array([g['box'] for g in groups],np.float32).reshape(-1,4)
        geometry[:,-1]=relative_geometry(boxes,baseline,np.zeros(k))
        indices=history_indices(groups,self.snapshots)
        for i in range(k):
            for t in range(HISTORY):
                row=int(indices[i,t])
                if row<0:continue
                old=self.records[row];assert old['frame']<frame
                f[i,t]=old['features'];m[i,t]=old['metadata'];valid[i,t]=True
                geometry[i,t]=relative_geometry(old['box'],baseline,frame-old['frame'])
        return dict(features=f,metadata=m,geometry=geometry,history_valid=valid)
    def observe(self,groups,features,metadata,frame):
        assert frame>self.last_frame;self.last_frame=frame
        if frame%STRIDE:return
        snapshot=dict(frame=frame,groups=[])
        for i,g in enumerate(groups):
            row=self.next_row;self.next_row+=1
            snapshot['groups'].append(dict(box=list(g['box']),member_ids=list(g['member_ids']),row=row))
            self.records[row]=dict(frame=frame,features=features[i].copy(),metadata=metadata[i].copy(),box=list(g['box']))
        self.snapshots.append(snapshot);self.snapshots=self.snapshots[-HISTORY:]
        needed={g['row'] for s in self.snapshots for g in s['groups']}
        self.records={key:value for key,value in self.records.items() if key in needed}
