import supervision as sv

ds_train = sv.DetectionDataset.from_yolo(
    images_directory_path=f'./my-trafficsign-1/train/images',
    annotations_directory_path=f'./my-trafficsign-1/train/labels',
    data_yaml_path=f'./my-trafficsign-1/train/data.yaml'
)
ds_valid = sv.DetectionDataset.from_yolo(
    images_directory_path=f'./my-trafficsign-1/valid/images',
    annotations_directory_path=f'./my-trafficsign-1/valid/labels',
    data_yaml_path=f'./my-trafficsign-1/valid/data.yaml'
)
ds_test = sv.DetectionDataset.from_yolo(
    images_directory_path=f'./my-trafficsign-1/test/images',
    annotations_directory_path=f'./my-trafficsign-1/test/labels',
    data_yaml_path=f'./my-trafficsign-1/test/data.yaml'
)

# 遍历方式1
for image_path, image, annotations in ds_train:
    print(f"Processing ds_train image: {image_path}")

# 遍历方式2
for idx in range(len(ds_valid)):
    image_path, image, annotations = ds_valid[idx]
    print(f"Processing ds_valid image: {image_path}")