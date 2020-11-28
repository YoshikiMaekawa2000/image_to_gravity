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

class FineTuner(trainer_mod.Trainer):
    def __init__(self,  #overwrite
            method_name,
            train_dataset, val_dataset,
            net, weights_path, criterion,
            optimizer_name, lr_cnn, lr_fc,
            batch_size, num_epochs):
        self.setRandomCondition()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device = ", self.device)
        self.dataloaders_dict = self.getDataloader(train_dataset, val_dataset, batch_size)
        self.net = self.getSetNetwork(net, weights_path)
        self.criterion = criterion
        self.optimizer = self.getOptimizer(optimizer_name, lr_cnn, lr_fc)
        self.num_epochs = num_epochs
        self.str_hyperparameter  = self.getStrHyperparameter(method_name, train_dataset, optimizer_name, lr_cnn, lr_fc, batch_size)

    def getSetNetwork(self, net, weights_path): #overwrite
        print(net)
        net.to(self.device)
        ## load
        if torch.cuda.is_available():
            loaded_weights = torch.load(weights_path)
            print("Loaded [GPU -> GPU]: ", weights_path)
        else:
            loaded_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("Loaded [GPU -> CPU]: ", weights_path)
        net.load_state_dict(loaded_weights)
        return net

    def getStrHyperparameter(self, method_name, dataset, optimizer_name, lr_cnn, lr_fc, batch_size):    #overwrite
        str_hyperparameter = method_name \
            + str(len(self.dataloaders_dict["train"].dataset)) + "tune" \
            + str(len(self.dataloaders_dict["val"].dataset)) + "val" \
            + str(dataset.transform.resize) + "resize" \
            + str(dataset.transform.mean[0]) + "mean" \
            + str(dataset.transform.std[0]) + "std" \
            + optimizer_name \
            + str(lr_cnn) + "lrcnn" \
            + str(lr_fc) + "lrfc" \
            + str(batch_size) + "batch" \
            + str(self.num_epochs) + "epoch"
        print("str_hyperparameter = ", str_hyperparameter)
        return str_hyperparameter

def main():
    ## hyperparameters
    method_name = "regression"
    train_rootpath = "../../../dataset_image_to_gravity/AirSim/1cam/train"
    val_rootpath = "../../../dataset_image_to_gravity/AirSim/1cam/val"
    csv_name = "imu_camera.csv"
    resize = 112
    mean_element = 0.5
    std_element = 0.5
    hor_fov_deg = 70
    optimizer_name = "Adam"  #"SGD" or "Adam"
    lr_cnn = 1e-6
    lr_fc = 1e-5
    batch_size = 100
    num_epochs = 50
    weights_path = "../../weights/regression.pth"
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
    net = network_mod.Network(resize, dim_fc_out=3, use_pretrained_vgg=False)
    ## criterion
    criterion = nn.MSELoss()
    ## train
    fine_tuner = FineTuner(
        method_name,
        train_dataset, val_dataset,
        net, weights_path, criterion,
        optimizer_name, lr_cnn, lr_fc,
        batch_size, num_epochs
    )
    fine_tuner.train()

if __name__ == '__main__':
    main()
