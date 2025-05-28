import supervision as sv

ds_train = sv.DetectionDataset.from_yolo(
    images_directory_path=f'./trafficsign-4/train/images',
    annotations_directory_path=f'./trafficsign-4/train/labels',
    data_yaml_path=f'./trafficsign-4/data.yaml'
)

print(ds_train.classes)
print(len(ds_train)) #输出训练集的样本数量，即图片数量