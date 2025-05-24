# 检测小物体需将模型从yolov8n.pt换成yolov8x.pt
# YOLOv8n.pt​​（Nano）：是系列中最轻量级的模型，参数量仅 ​​3.2M​​，网络层数较少，适合资源受限的环境（如移动端或嵌入式设备）
# ​​YOLOv8x.pt​​（Extra Large）：是最大的版本，参数量高达 ​​68.2M​​，网络结构更深且复杂，能够捕捉更细粒度的特征，但需要更强的计算资源支持

import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
image = cv2.imread("small_objects.jpeg")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# 保存处理后的图像（文件路径可自定义）
cv2.imwrite("detect_small_objects.png", annotated_image)

cv2.imshow("YOLOv8", annotated_image)
cv2.waitKey(0)