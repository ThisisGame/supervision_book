import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")
image = cv2.imread("small_objects.jpeg")

# 使用分割着色模型(yolov8x-seg)进行小物体检测，InferenceSlicer也可以执行分割着色任务。

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(callback = callback)
detections = slicer(image)

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = mask_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# 保存处理后的图像（文件路径可自定义）
cv2.imwrite("detect_small_objects_segmentation.png", annotated_image)

cv2.imshow("YOLOv8", annotated_image)
cv2.waitKey(0)