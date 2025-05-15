## 过滤

可以过滤识别的物体类型、也可以选择指定图像区域、也可以按物体像素占比来过滤，或者混合多种条件过滤。

可以用来节省性能开销。


### 按物体类型过滤

下面代码过滤出人，进行标注。

```python
#file:files\supervision\detect_and_annotate_by_specific_class.py

import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = cv2.imread("supervision-detection-by-specific.png")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.class_id == 0]#只处理特定类别，这里是0类，即person类

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
```

![](../../imgs/supervision/supervision-detection-by-specific-0-person.png)

### 按多个类型过滤

下面指定多种类型进行过滤。

```python
#file:files\supervision\detect_and_annotate_by_set_of_classes.py

import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")
image = cv2.imread("supervision-detection-by-specific.png")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

selected_classes = [0, 2, 3]
detections = detections[np.isin(detections.class_id, selected_classes)]#按selected_classes里的多个类别筛选

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
```

![](../../imgs/supervision/supervision-detection-by-set-of-classed.png)


### 按可信度过滤

按照可信度过滤，例如只对可信度大于0.5的进行标注，可以排除一些干扰。

```python
#file:files\supervision\detect_and_annotate_by_confidence.py

import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")
image = cv2.imread("supervision-detection-by-specific.png")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.confidence > 0.5]#只对可信度大于0.5的进行标注，可以排除一些干扰。

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
```

![](../../imgs/supervision/detect_and_annotate_by_confidence.png)


