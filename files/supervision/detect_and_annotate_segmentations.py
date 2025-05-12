import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
image = cv2.imread("highway_traffic.png")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

annotated_image = mask_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

cv2.imshow("YOLOv8", annotated_image)
cv2.waitKey(0)