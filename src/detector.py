import numpy as np
import torch
from ultralytics import YOLO

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

np.random.seed(42)
_COLOURS = np.random.randint(60,220,(len(COCO_CLASSES),3),dtype=np.uint8)

def class_name(cls_id):
    return COCO_CLASSES[cls_id] if 0<=cls_id<len(COCO_CLASSES) else f"cls{cls_id}"

def class_color(cls_id):
    idx=cls_id%len(_COLOURS)
    return int(_COLOURS[idx,0]),int(_COLOURS[idx,1]),int(_COLOURS[idx,2])

class Detector:
    def __init__(self,model_name="yolov8n.pt",conf=0.40,nms_iou=0.45,
                 device=None,classes=None,input_size=640):
        self.conf=conf; self.nms_iou=nms_iou
        self.classes=set(classes) if classes else None
        self.input_size=input_size
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Detector] Loading {model_name} on {self.device} ...")
        self.model=YOLO(model_name)
        self.model.to(self.device)
        print("[Detector] Ready.")

    def detect(self,frame):
        results=self.model.predict(
            frame,imgsz=self.input_size,
            conf=self.conf,iou=self.nms_iou,
            device=self.device,verbose=False)
        rows=[]
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                cls_id=int(box.cls[0])
                if self.classes and cls_id not in self.classes: continue
                x1,y1,x2,y2=box.xyxy[0].tolist()
                rows.append([x1,y1,x2,y2,float(box.conf[0]),cls_id])
        return np.array(rows,dtype=float) if rows else np.empty((0,6))