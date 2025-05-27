## 给检测到的对象添加方框

import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
box_annotator = sv.BoundingBoxAnnotator()

# 使用YOLOv8模型检测每一帧图片中的对象，并在每个检测到的对象周围添加方框。
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    return box_annotator.annotate(frame.copy(), detections=detections)

# 将视频拆分为一帧一帧的图片，调用callback函数处理每一帧，然后合成视频。
sv.process_video(
    source_path="people-walking.mp4",
    target_path="detect_video_objects.mp4",
    callback=callback
)