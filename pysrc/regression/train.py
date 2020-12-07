from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import trainer_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import network_mod

def main():
    ## hyperparameters
    method_name = "regression"
    train_rootpath = "../../../dataset_image_to_gravity/AirSim/1cam/train"
    val_rootpath = "../../../dataset_image_to_gravity/AirSim/1cam/val"
    csv_name = "imu_camera.csv"
    resize = 224
    mean_element = 0.5
    std_element = 0.5
    hor_fov_deg = 70
    optimizer_name = "Adam"  #"SGD" or "Adam"
    lr_cnn = 1e-5
    lr_fc = 1e-4
    batch_size = 50
    num_epochs = 50
    ## dataset
    train_dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(train_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element]),
            hor_fov_deg=hor_fov_deg
        ),
        phase="train"
    )
    val_dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(val_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element]),
            hor_fov_deg=hor_fov_deg
        ),
        phase="val"
    )
    ## network
    net = network_mod.Network(resize, dim_fc_out=3, use_pretrained_vgg=True)
    ## criterion
    criterion = nn.MSELoss()
    ## train
    trainer = trainer_mod.Trainer(
        method_name,
        train_dataset, val_dataset,
        net, criterion,
        optimizer_name, lr_cnn, lr_fc,
        batch_size, num_epochs
    )
    trainer.train()

if __name__ == '__main__':
    main()
