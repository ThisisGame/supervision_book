import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")
image = cv2.imread("supervision-detection-by-specific.png")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.area > 1000]#只对像素大于1000的物体进行标注。这样可以过滤掉一些小物体。

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

# 保存处理后的图像（文件路径可自定义）
cv2.imwrite("detect_and_annotate_by_area.png", annotated_image)

cv2.imshow("YOLOv8", annotated_image)
cv2.waitKey(0)