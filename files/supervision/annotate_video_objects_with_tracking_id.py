## 给检测到的对象添加方框并显示Tracking ID，Tracking ID是一个唯一的标识符，用于跟踪视频中每个对象的运动轨迹。

import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 使用YOLOv8模型检测每一帧图片中的对象，并在每个检测到的对象周围添加方框。
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)# 更新检测结果以包含跟踪ID

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)#显示Tracking ID

# 将视频拆分为一帧一帧的图片，调用callback函数处理每一帧，然后合成视频。
sv.process_video(
    source_path="people-walking.mp4",
    target_path="annotate_video_objects_with_tracking_id.mp4",
    callback=callback
)