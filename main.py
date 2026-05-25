import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sort_tracker import Sort
from src.detector import Detector, class_name, class_color
from src.visualizer import FPSCounter, TrailManager, draw_overlay

def run():
    SOURCE     = 0
    MODEL      = "yolov8n.pt"
    CONFIDENCE = 0.40
    MAX_AGE    = 30
    MIN_HITS   = 3
    TRAIL_LEN  = 40

    print("Loading detector...")
    detector    = Detector(model_name=MODEL, conf=CONFIDENCE)
    tracker     = Sort(max_age=MAX_AGE, min_hits=MIN_HITS)
    trails      = TrailManager(max_length=TRAIL_LEN)
    fps_counter = FPSCounter()

    print("Opening webcam...")
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        print("Try changing SOURCE = 0 to SOURCE = 1 in main.py")
        input("Press Enter to close...")
        return

    print("Running! Press Q to quit, P to pause, S to screenshot.")
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame - stream ended.")
            break

        frame_num += 1
        detections = detector.detect(frame)
        det_sort   = detections[:, :5] if len(detections) else np.empty((0,5))
        tracks     = tracker.update(det_sort)
        fps        = fps_counter.tick()
        out        = draw_overlay(frame, detections, tracks,
                                  class_name, class_color,
                                  trails, fps, frame_num)

        cv2.imshow("Object Detection & Tracking [Q=quit]", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
        elif key == ord('s'):
            os.makedirs("output", exist_ok=True)
            cv2.imwrite(f"output/screenshot_{frame_num:05d}.jpg", out)
            print("Screenshot saved!")

        if frame_num % 30 == 0:
            print(f"Frame {frame_num} | FPS {fps:.1f} | Det {len(detections)} | Tracks {len(tracks)}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("ERROR:", e)
        import traceback
        traceback.print_exc()
    input("Press Enter to close...")