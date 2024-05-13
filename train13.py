import os
import time
# HOME = os.getcwd()
from typing import List
import cv2
from IPython import display
display.clear_output()
import sys
sys.path.append(r"C:\Users\Divyansh\Desktop\trm\try\ByteTrack")
import yolox
print("yolox.__version__:", yolox.__version__)
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
from IPython import display
display.clear_output()
import supervision
print("supervision.__version__:", supervision.__version__)
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
import numpy as np
track = [0,0,0,0]

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id
    return tracker_ids

def counter(detections):
    path1 = [0,0,0,0]
    for _, confidence, class_id, tracker_id in detections:
            if class_id == 2:
                path1[0] = path1[0] + 1
            elif class_id == 3:
                path1[1] = path1[1] + 1
            elif class_id == 5:
                path1[2] = path1[2] + 1
            elif class_id == 7:
                path1[3] = path1[3] + 1
    return path1

MODEL = "yolov8x.pt"
from ultralytics import YOLO
model = YOLO(MODEL)
model.fuse()
CLASS_NAMES_DICT = model.model.names
CLASS_ID = [2, 3, 5, 7]
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)

def detection_and_label(results):
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    return detections,labels

def totaltime(arr):
    time = 0
    #car, moterbikes, bus, truck
    t = [3,2,6,5]
    for i,it in enumerate(arr):
        time = it*t[i] + time
    time = max(time,5)
    time = min(time,150)
    return time


cap = cv2.VideoCapture(r"C:\Users\Divyansh\Desktop\trm\ultralytics\highway.mp4")
# cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(r"C:\Users\Divyansh\Desktop\trm\ultralytics\highway.mp4")
cap2 = cv2.VideoCapture(r"C:\Users\Divyansh\Desktop\trm\ultralytics\highway.mp4")
cap3 = cv2.VideoCapture(r"C:\Users\Divyansh\Desktop\trm\ultralytics\highway.mp4")
while(cap.isOpened or cap1.isOpened or cap2.isOpened or cap3.isOpened):

    ret,frame=cap.read()
    if ret:
        results = model(frame)
        detections,labels = detection_and_label(results)
        track1 = counter(detections)
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        frame = cv2.resize(frame,(640,640))
        track[0] = totaltime(track1)
        print(f"track1 time :- {str(track[0])}")
    
    ret1,frame1=cap1.read()
    if ret1:
        results = model(frame1)
        detections,labels = detection_and_label(results)
        track2 = counter(detections)
        frame1 = box_annotator.annotate(frame=frame1, detections=detections, labels=labels)
        frame1 = cv2.resize(frame1,(640,640))
        track[1] = totaltime(track2)
        print(f"track2 time :- {str(track[1])}")
    
    ret2,frame2=cap2.read()
    if ret2:
    # frame2 = cv2.imread("i1.jpg")
        results = model(frame2)
        detections,labels = detection_and_label(results)
        track3 = counter(detections)
        frame2 = box_annotator.annotate(frame=frame2, detections=detections, labels=labels)
        frame2 = cv2.resize(frame2,(640,640))
        track[2] = totaltime(track3)
        print(f"track3 time :- {str(track[2])}")

    ret3,frame3=cap3.read()
    if ret3:
    # frame3 = cv2.imread("IMG_20240419_183706.jpg")
        results = model(frame3)
        detections,labels = detection_and_label(results)
        track4 = counter(detections)
        frame3 = box_annotator.annotate(frame=frame3, detections=detections, labels=labels)
        frame3 = cv2.resize(frame3,(640,640))
        track[3] = totaltime(track4)
        # cv2.imshow("track",frame3)
        print(f"track4 time :- {str(track[3])}")

    k = cv2.waitKey(30) & 0xFF
    if k == ord('m'):
        break
    else:
        time.sleep(5)
        for i,_ in enumerate(track):
            track[i] = max(track[i],0)

cap.release()
cv2.destroyAllWindows()