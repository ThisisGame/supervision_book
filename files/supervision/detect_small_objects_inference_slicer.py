import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
image = cv2.imread("small_objects.jpeg")

# 对每个切成小块的图像进行推理
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(callback = callback)
detections = slicer(image)# 使用InferenceSlicer进行小物体检测

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# 保存处理后的图像（文件路径可自定义）
cv2.imwrite("detect_small_objects_inference_slicer.png", annotated_image)

cv2.imshow("YOLOv8", annotated_image)
cv2.waitKey(0)