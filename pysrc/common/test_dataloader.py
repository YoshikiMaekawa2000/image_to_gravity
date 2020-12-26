import numpy as np
import matplotlib.pyplot as plt

import torch

import make_datalist_mod
import data_transform_mod
import dataset_mod

def show_inputs(inputs):
    h = 5
    w = 5
    plt.figure()
    for i, tensor in enumerate(inputs):
        if i < h*w:
            img = tensor.numpy().transpose((1, 2, 0))
            img = np.clip(img, 0, 1)
            plt.subplot(h, w, i+1)
            plt.imshow(img)
    plt.show()

## list
list_train_rootpath = ["../../../dataset_image_to_gravity/AirSim/1cam/train"]
list_val_rootpath = ["../../../dataset_image_to_gravity/AirSim/1cam/val"]
csv_name = "imu_camera.csv"
train_list = make_datalist_mod.makeDataList(list_train_rootpath, csv_name)
val_list = make_datalist_mod.makeDataList(list_val_rootpath, csv_name)

## trans param
resize = 224
mean = ([0.5, 0.5, 0.5])
std = ([0.5, 0.5, 0.5])

## dataset
train_dataset = dataset_mod.OriginalDataset(
    data_list=train_list,
    transform=data_transform_mod.DataTransform(resize, mean, std),
    phase="train"
)
val_dataset = dataset_mod.OriginalDataset(
    data_list=val_list,
    transform=data_transform_mod.DataTransform(resize, mean, std),
    phase="val"
)

# dataloader
batch_size = 10

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

batch_iterator = iter(dataloaders_dict["train"])
# batch_iterator = iter(dataloaders_dict["val"])
inputs, labels = next(batch_iterator)

# print("inputs = ", inputs)
print("inputs.size() = ", inputs.size())
print("labels = ", labels)
print("labels[0] = ", labels[0])
print("labels.size() = ", labels.size())
show_inputs(inputs)
