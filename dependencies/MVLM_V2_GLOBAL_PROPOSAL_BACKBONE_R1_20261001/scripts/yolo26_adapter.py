#!/usr/bin/env python3
from ultralytics import YOLO
class YOLO26Detector:
 def __init__(self,weights,imgsz=1280,conf=0.001,iou=0.60,max_det=100,device=0):
  self.model=YOLO(str(weights)); self.imgsz=imgsz; self.conf=conf; self.iou=iou; self.max_det=max_det; self.device=device
 def infer(self,bgr):
  rr=self.model.predict(source=bgr,imgsz=self.imgsz,conf=self.conf,iou=self.iou,max_det=self.max_det,device=self.device,verbose=False)
  r=rr[0]; out=[]
  if r.boxes is None: return out
  xyxy=r.boxes.xyxy.detach().cpu().numpy(); conf=r.boxes.conf.detach().cpu().numpy(); cls=r.boxes.cls.detach().cpu().numpy()
  for b,s,c in zip(xyxy,conf,cls):
   x1,y1,x2,y2=map(float,b); out.append({'box':[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],'score':float(s),'cls':int(c)})
  out.sort(key=lambda z:z['score'],reverse=True); return out
