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
import criterion_mod

class Trainer(trainer_mod.Trainer):
    def saveGraph(self, record_loss_train, record_loss_val):    #overwrite
        graph = plt.figure()
        plt.plot(range(len(record_loss_train)), record_loss_train, label="Training")
        plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss [m/s^2]")
        plt.title("loss: train=" + str(record_loss_train[-1]) + ", val=" + str(record_loss_val[-1]))
        graph.savefig("../../graph/" + self.str_hyperparameter + ".jpg")
        plt.show()

def main():
    ## hyperparameters
    method_name = "mle"
    list_train_rootpath = ["../../../dataset_image_to_gravity/AirSim/1cam/train"]
    list_val_rootpath = ["../../../dataset_image_to_gravity/AirSim/1cam/val"]
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
        data_list=make_datalist_mod.makeDataList(list_train_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element]),
            hor_fov_deg=hor_fov_deg
        ),
        phase="train"
    )
    val_dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(list_val_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element]),
            hor_fov_deg=hor_fov_deg
        ),
        phase="val"
    )
    ## network
    net = network_mod.Network(resize, list_dim_fc_out=[100, 18, 9], dropout_rate=0.1, use_pretrained_vgg=True)
    ## criterion
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = criterion_mod.Criterion(device)
    ## train
    trainer = Trainer(
        method_name,
        train_dataset, val_dataset,
        net, criterion,
        optimizer_name, lr_cnn, lr_fc,
        batch_size, num_epochs
    )
    trainer.train()

if __name__ == '__main__':
    main()
