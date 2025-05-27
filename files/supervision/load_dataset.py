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

print(ds_train.classes)
print(len(ds_train), len(ds_valid), len(ds_test))