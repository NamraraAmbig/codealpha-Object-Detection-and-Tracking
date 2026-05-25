import cv2
import numpy as np
from collections import deque
import time

class FPSCounter:
    def __init__(self,window=30):
        self._t=deque(maxlen=window)
    def tick(self):
        self._t.append(time.perf_counter())
        if len(self._t)<2: return 0.0
        return (len(self._t)-1)/(self._t[-1]-self._t[0])

class TrailManager:
    def __init__(self,max_length=30):
        self.trails={}; self.max_length=max_length
    def update(self,tid,cx,cy):
        if tid not in self.trails:
            self.trails[tid]=deque(maxlen=self.max_length)
        self.trails[tid].append((cx,cy))
    def draw(self,frame,tid,color):
        pts=list(self.trails.get(tid,[]))
        for i in range(1,len(pts)):
            alpha=i/len(pts)
            cv2.line(frame,pts[i-1],pts[i],color,max(1,int(alpha*2)),cv2.LINE_AA)

def draw_box(frame,x1,y1,x2,y2,color,thickness=2,cl=15):
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
    for pts in [
        [(x1,y1+cl),(x1,y1),(x1+cl,y1)],
        [(x2-cl,y1),(x2,y1),(x2,y1+cl)],
        [(x1,y2-cl),(x1,y2),(x1+cl,y2)],
        [(x2-cl,y2),(x2,y2),(x2,y2-cl)]]:
        cv2.polylines(frame,[np.array(pts)],False,color,thickness,cv2.LINE_AA)

def draw_label(frame,text,x1,y1,color):
    font=cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_=cv2.getTextSize(text,font,0.55,1)
    pad=4
    cv2.rectangle(frame,(x1,max(0,y1-th-2*pad)),(x1+tw+2*pad,y1),color,-1)
    cv2.putText(frame,text,(x1+pad,y1-pad),font,0.55,(255,255,255),1,cv2.LINE_AA)

def draw_hud(frame,fps,frame_num,n_det,n_trk):
    lines=[f"FPS:        {fps:5.1f}",f"Frame:      {frame_num:5d}",
           f"Detections: {n_det:5d}",f"Tracks:     {n_trk:5d}"]
    overlay=frame.copy()
    cv2.rectangle(overlay,(10,10),(210,100),(20,20,20),-1)
    cv2.addWeighted(overlay,0.65,frame,0.35,0,frame)
    for i,line in enumerate(lines):
        cv2.putText(frame,line,(18,30+i*18),cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(200,200,200),1,cv2.LINE_AA)

def draw_overlay(frame,detections,tracks,name_fn,color_fn,trails,fps,frame_num):
    out=frame.copy()
    for trk in tracks:
        x1,y1,x2,y2,tid=int(trk[0]),int(trk[1]),int(trk[2]),int(trk[3]),int(trk[4])
        cls_id=0; best=0.0
        for det in detections:
            dx1,dy1,dx2,dy2=det[:4]
            inter=max(0,min(x2,dx2)-max(x1,dx1))*max(0,min(y2,dy2)-max(y1,dy1))
            union=(x2-x1)*(y2-y1)+(dx2-dx1)*(dy2-dy1)-inter
            v=inter/union if union>0 else 0
            if v>best: best=v; cls_id=int(det[5])
        color=color_fn(cls_id)
        cx,cy=(x1+x2)//2,(y1+y2)//2
        trails.update(tid,cx,cy); trails.draw(out,tid,color)
        draw_box(out,x1,y1,x2,y2,color)
        draw_label(out,f"ID:{tid} {name_fn(cls_id)}",x1,y1,color)
    draw_hud(out,fps,frame_num,len(detections),len(tracks))
    return out