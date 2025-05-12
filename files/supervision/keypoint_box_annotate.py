## 检测人物骨骼、关节点，并绘制骨架，然后绘制方框

import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")
edge_annotator = sv.EdgeAnnotator()
vertex_annotator = sv.VertexAnnotator()
box_annotator = sv.BoxAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    key_points = sv.KeyPoints.from_ultralytics(results)
    detections = key_points.as_detections()

    annotated_frame = edge_annotator.annotate(
        frame.copy(), key_points=key_points)
    annotated_frame = vertex_annotator.annotate(
        annotated_frame, key_points=key_points)
    return box_annotator.annotate(
        annotated_frame, detections=detections)

sv.process_video(
    source_path="skiing.mp4",
    target_path="result.mp4",
    callback=callback
)