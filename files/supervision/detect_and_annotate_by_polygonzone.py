import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")
image = cv2.imread("supervision-detection-by-specific.png")

results = model(image)[0]

# 获取检测结果
detections = sv.Detections.from_ultralytics(results)

# 定义多边形区域的顶点（示例为一个四边形）
polygon_vertices = [(100, 100), (500, 100), (500, 400), (100, 400)]
zone = sv.PolygonZone(polygon=polygon_vertices, frame_resolution_wh=(800, 600))

# 生成掩码（True表示检测框在区域内）
mask = zone.trigger(detections=detections)

# 应用过滤，仅保留区域内的检测结果
detections = detections[mask]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

cv2.imshow("YOLOv8", annotated_image)
cv2.waitKey(0)