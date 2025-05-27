import supervision as sv

ds = sv.DetectionDataset.from_yolo(
    images_directory_path=f'./trafficsign-4/train/images',
    annotations_directory_path=f'./trafficsign-4/train/labels',
    data_yaml_path=f'./trafficsign-4/data.yaml'
)

ds_train, ds = ds.split(split_ratio=0.8, shuffle=True)
ds_valid, ds_test = ds.split(split_ratio=0.5, shuffle=True)

print(len(ds_train), len(ds_valid), len(ds_test))

# Save the datasets to disk
ds_train.as_yolo(
    images_directory_path='./my-trafficsign-1/train/images',
    annotations_directory_path='./my-trafficsign-1/train/labels',
    data_yaml_path='./my-trafficsign-1/train/data.yaml'
)
ds_valid.as_yolo(
    images_directory_path='./my-trafficsign-1/valid/images',
    annotations_directory_path='./my-trafficsign-1/valid/labels',
    data_yaml_path='./my-trafficsign-1/valid/data.yaml'
)
ds_test.as_yolo(
    images_directory_path='./my-trafficsign-1/test/images',
    annotations_directory_path='./my-trafficsign-1/test/labels',
    data_yaml_path='./my-trafficsign-1/test/data.yaml'
)