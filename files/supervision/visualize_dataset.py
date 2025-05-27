import supervision as sv
import matplotlib.pyplot as plt

ds_train = sv.DetectionDataset.from_yolo(
    images_directory_path=f'./my-trafficsign-1/train/images',
    annotations_directory_path=f'./my-trafficsign-1/train/labels',
    data_yaml_path=f'./my-trafficsign-1/train/data.yaml'
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_images = []
for i in range(16):
    image_path, image, annotations = ds_train[i] # 获取第 i 张图片和对应的注释(annotations)
    print(f"i: {i}, image_path: {image_path}, annotations: {annotations}")

    labels = [ds_train.classes[class_id] for class_id in annotations.class_id]

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
    annotated_images.append(annotated_image)

print(f"Total annotated images: {len(annotated_images)}")

# 创建一个4x4的网格来展示16张图片
grid = sv.create_tiles(
    annotated_images,
    grid_size=(4, 4),
    single_tile_size=(400, 400),
    tile_padding_color=sv.Color.WHITE,
    tile_margin_color=sv.Color.WHITE
)

plt.imshow(grid)
plt.axis('off')
plt.savefig(
    './visualize_dataset.jpg',         # 文件名（支持.png/.jpg/.pdf等格式）
    dpi=300,                       # 分辨率（默认100）
    bbox_inches='tight',           # 去除多余白边
    pad_inches=0                  # 内边距控制
)
plt.show()