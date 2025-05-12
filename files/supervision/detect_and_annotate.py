import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = cv2.imread("highway_traffic.png")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

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