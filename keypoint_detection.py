## 检测人物骨骼、关节点，并绘制骨架

import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")
edge_annotator = sv.EdgeAnnotator()
vertex_annotator = sv.VertexAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    key_points = sv.KeyPoints.from_ultralytics(results)

    annotated_frame = edge_annotator.annotate(
        frame.copy(), key_points=key_points)
    return vertex_annotator.annotate(
        annotated_frame, key_points=key_points)

sv.process_video(
    source_path="skiing.mp4",
    target_path="result.mp4",
    callback=callback
)