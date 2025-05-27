# 1. 定义增强操作（前置步骤）
import albumentations as A

augmentation = A.Compose([
    A.Perspective(p=0.1),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5)
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"]))

# 2. 应用增强到单张图像（你提供的代码）
import numpy as np
import supervision as sv
from dataclasses import replace

# 加载数据
ds_train = sv.DetectionDataset.from_yolo(
    images_directory_path=f'./my-trafficsign-1/train/images',
    annotations_directory_path=f'./my-trafficsign-1/train/labels',
    data_yaml_path=f'./my-trafficsign-1/train/data.yaml'
)
_, image, annotations = ds_train[0]

# 调用增强操作
output = augmentation(
    image=image,
    bboxes=annotations.xyxy,
    category=annotations.class_id
)

# 提取增强结果
augmented_image = output["image"]
augmented_bboxes = np.array(output["bboxes"])
augmented_class_ids = np.array(output["category"])

# 更新标注
augmented_annotations = replace(
    annotations,
    xyxy=augmented_bboxes,
    class_id=augmented_class_ids
)

# 定义标注工具
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 原始图像标注
original_labels = [ds_train.classes[class_id] for class_id in annotations.class_id]
original_annotated = box_annotator.annotate(image.copy(), annotations)
original_annotated = label_annotator.annotate(original_annotated, annotations, original_labels)

# 增强后图像标注
augmented_labels = [ds_train.classes[class_id] for class_id in augmented_class_ids]
augmented_annotated = box_annotator.annotate(augmented_image.copy(), augmented_annotations)
augmented_annotated = label_annotator.annotate(augmented_annotated, augmented_annotations, augmented_labels)

# 对比显示
sv.plot_images_grid(
    [original_annotated, augmented_annotated],
    grid_size=(1, 2),
    titles=["Original", "Augmented"]
)