import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")
image = cv2.imread("supervision-detection-by-specific.png")

# 计算图像的面积
height, width, channels = image.shape
image_area = height * width

results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

# detections.area表示检测框的面积
# detections.area / image_area表示检测框面积占图像面积的比例
# detections[(detections.area / image_area) < 0.8]表示只保留面积占图像面积小于0.8的检测框
detections = detections[(detections.area / image_area) < 0.8]

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