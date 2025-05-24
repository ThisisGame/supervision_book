import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
image = cv2.imread("small_objects.jpeg")
results = model(image, imgsz=640 * 4)[0] # 将图像放大4倍
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# 保存处理后的图像（文件路径可自定义）
cv2.imwrite("detect_small_objects_up_resolution.png", annotated_image)

cv2.imshow("YOLOv8", annotated_image)
cv2.waitKey(0)