import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(a, b):
    xx1, yy1 = max(a[0],b[0]), max(a[1],b[1])
    xx2, yy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xx2-xx1)*max(0,yy2-yy1)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.0

def box_to_z(box):
    w=box[2]-box[0]; h=box[3]-box[1]
    return np.array([box[0]+w/2,box[1]+h/2,w*h,w/max(h,1)]).reshape(4,1)

def z_to_box(x):
    w=np.sqrt(max(x[2]*x[3],0)); h=x[2]/max(w,1)
    return np.array([x[0]-w/2,x[1]-h/2,x[0]+w/2,x[1]+h/2]).reshape(1,4)

class KalmanBoxTracker:
    count=0
    def __init__(self,bbox):
        kf=KalmanFilter(dim_x=7,dim_z=4)
        kf.F=np.eye(7); kf.F[0,4]=kf.F[1,5]=kf.F[2,6]=1
        kf.H=np.eye(4,7)
        kf.R[2:,2:]*=10; kf.P[4:,4:]*=1000; kf.P*=10
        kf.Q[-1,-1]*=0.01; kf.Q[4:,4:]*=0.01
        kf.x[:4]=box_to_z(bbox)
        self.kf=kf
        self.id=KalmanBoxTracker.count; KalmanBoxTracker.count+=1
        self.time_since_update=0; self.hit_streak=0; self.hits=0; self.age=0
    def predict(self):
        if self.kf.x[6]+self.kf.x[2]<=0: self.kf.x[6]=0
        self.kf.predict(); self.age+=1
        if self.time_since_update>0: self.hit_streak=0
        self.time_since_update+=1
        return z_to_box(self.kf.x)
    def update(self,bbox):
        self.time_since_update=0; self.hits+=1; self.hit_streak+=1
        self.kf.update(box_to_z(bbox))
    def get_state(self):
        return z_to_box(self.kf.x)

def associate(detections, trackers, iou_thresh=0.3):
    if len(trackers)==0:
        return np.empty((0,2),int), np.arange(len(detections)), np.empty(0,int)
    if len(detections)==0:
        return np.empty((0,2),int), np.empty(0,int), np.arange(len(trackers))

    n_det = len(detections)
    n_trk = len(trackers)
    C = np.zeros((n_det, n_trk))
    for d in range(n_det):
        for t in range(n_trk):
            C[d,t] = iou(detections[d], trackers[t])

    ri, ci = linear_sum_assignment(-C)
    matched, u_dets, u_trks = [], [], []
    for d in range(n_det):
        if d not in ri: u_dets.append(d)
    for t in range(n_trk):
        if t not in ci: u_trks.append(t)
    for r,c in zip(ri,ci):
        if C[r,c] >= iou_thresh: matched.append([r,c])
        else: u_dets.append(r); u_trks.append(c)
    return (np.array(matched) if matched else np.empty((0,2),int),
            np.array(u_dets), np.array(u_trks))

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age=max_age; self.min_hits=min_hits
        self.iou_threshold=iou_threshold
        self.trackers=[]; self.frame_count=0
        KalmanBoxTracker.count=0

    def update(self, dets=np.empty((0,5))):
        self.frame_count += 1
        predicted = np.array([t.predict()[0] for t in self.trackers]) \
                    if self.trackers else np.empty((0,4))
        matched, u_dets, u_trks = associate(
            dets[:,:4] if len(dets) else np.empty((0,4)),
            predicted, self.iou_threshold)
        for d,t in matched:
            self.trackers[t].update(dets[d,:4])
        for d in u_dets:
            self.trackers.append(KalmanBoxTracker(dets[d,:4]))
        out = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            confirmed = trk.hit_streak>=self.min_hits or self.frame_count<=self.min_hits
            if trk.time_since_update<1 and confirmed:
                out.append([*trk.get_state()[0], trk.id+1])
            if trk.time_since_update>self.max_age:
                self.trackers.pop(i)
        return np.array(out) if out else np.empty((0,5))