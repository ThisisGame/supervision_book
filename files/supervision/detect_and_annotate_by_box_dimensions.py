import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt").to('cuda')  # 将模型加载到GPU
image = cv2.imread("supervision-detection-by-specific.png")

results = model(image , device=1)[0]
detections = sv.Detections.from_ultralytics(results)

# 输出检测结果
print(f"检测到的目标数量: {len(detections)}")

# 计算检测框的宽度和高度
w = detections.xyxy[:, 2] - detections.xyxy[:, 0]
h = detections.xyxy[:, 3] - detections.xyxy[:, 1]

# 只保留宽度和高度都大于200的检测框
detections = detections[(w > 200) & (h > 200)]

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
cv2.imwrite("detect_and_annotate_by_box_dimensions.png", annotated_image)

cv2.imshow("YOLOv8", annotated_image)
cv2.waitKey(0)