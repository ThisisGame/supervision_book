## 检测人物骨骼、关节点，并绘制骨架

import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")
edge_annotator = sv.EdgeAnnotator() #边的标注器
vertex_annotator = sv.VertexAnnotator() #点的标注器
box_annotator = sv.BoxAnnotator() #边框标注器

tracker = sv.ByteTrack()
trace_annotator = sv.TraceAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    key_points = sv.KeyPoints.from_ultralytics(results) #从YOLO获取识别到的关节点
    detections = key_points.as_detections() #将关节点转换为检测对象

    detections = tracker.update_with_detections(detections) #跟踪检测对象

    annotated_frame = edge_annotator.annotate(frame.copy(), key_points=key_points) #先绘制骨架
    annotated_frame = vertex_annotator.annotate(annotated_frame, key_points=key_points) #再绘制关节点
    annotated_frame = box_annotator.annotate(annotated_frame, detections=detections) # 绘制边框
    return trace_annotator.annotate(annotated_frame, detections=detections) # 绘制轨迹

sv.process_video(
    source_path="skiing.mp4",
    target_path="keypoint_detection_and_tracking.mp4",
    callback=callback
)