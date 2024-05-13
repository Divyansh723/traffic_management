import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO 
from ultralytics.utils.torch_utils import select_device
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import argparse
import supervision as sv


def parse_arguments() -> argparse.Namespace:
    parse = argparse.ArgumentParser(description = "yolov8 live")
    parse.add_argument(
        "webcam-resolution",
        default=[1200,720],
        nargs=2,
        type = int
    )
    args = parse.parse_args()
    return args

# args = parse_arguments()
# w,h = args.webcam_resolution
cap = cv2.VideoCapture(r"C:\Users\Divyansh\Desktop\trm\ultralytics\highway.mp4")
model = YOLO("yolov8n.pt")
CLASS_NAMES_DICT = model.model.names
CLASS_ID = [2, 3, 5, 7]
box_annotator = sv.BoxAnnotator(
    color=ColorPalette(), 
    thickness=2, 
    text_thickness=2, 
    text_scale=1
    )
# class_id1 = [2,3,5,7]

while(cap.isOpened()):
    ret, frame = cap.read()
    # results = model.predict(source=frame , show = False)
    results = model(frame)[0]
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

fun.release()
cv2.destroyAllWindows()