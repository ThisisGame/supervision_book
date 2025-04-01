## 检测人物骨骼、关节点，并绘制骨架，然后绘制方框，最后显示对象运动轨迹，应用平滑轨迹。

import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")
edge_annotator = sv.EdgeAnnotator()
vertex_annotator = sv.VertexAnnotator()
box_annotator = sv.BoxAnnotator()

tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()
trace_annotator = sv.TraceAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    key_points = sv.KeyPoints.from_ultralytics(results)
    detections = key_points.as_detections()
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    annotated_frame = edge_annotator.annotate(
        frame.copy(), key_points=key_points)
    annotated_frame = vertex_annotator.annotate(
        annotated_frame, key_points=key_points)
    annotated_frame = box_annotator.annotate(
        annotated_frame, detections=detections)
    return trace_annotator.annotate(
        annotated_frame, detections=detections)

sv.process_video(
    source_path="skiing.mp4",
    target_path="result.mp4",
    callback=callback
)