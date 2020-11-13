from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import make_datalist_mod
import data_transform_mod
import dataset_mod
import network_mod

class TrainModel:
    def __init__(self,
            method_name,
            train_rootpath, val_rootpath, csv_name,
            resize, mean_element, std_element,
            optimizer_name, lr_cnn, lr_fc,
            batch_size, num_epochs):
        self.setRandomCondition()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device = ", self.device)
        self.data_transform = self.getDataTransform(resize, mean_element, std_element)
        self.dataloaders_dict = self.getDataloader(train_rootpath, val_rootpath, csv_name, batch_size)
        self.net = self.getNetwork(resize)
        self.optimizer = self.getOptimizer(optimizer_name, lr_cnn, lr_fc)
        self.num_epochs = num_epochs
        self.str_hyperparameter  = self.getStrHyperparameter(method_name, resize, mean_element, std_element, optimizer_name, lr_cnn, lr_fc, batch_size)

    def setRandomCondition(self, keep_reproducibility=False):
        if keep_reproducibility:
            torch.manual_seed(1234)
            np.random.seed(1234)
            random.seed(1234)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def getDataTransform(self, resize, mean_element, std_element):
        mean = ([mean_element, mean_element, mean_element])
        std = ([std_element, std_element, std_element])
        data_transform = data_transform_mod.DataTransform(resize, mean, std)
        return data_transform

    def getDataloader(self, train_rootpath, val_rootpath, csv_name, batch_size):
        ## list
        train_list = make_datalist_mod.makeDataList(train_rootpath, csv_name)
        val_list = make_datalist_mod.makeDataList(val_rootpath, csv_name)
        ## dataset
        train_dataset = dataset_mod.OriginalDataset(
            data_list=train_list,
            transform=self.data_transform,
            phase="train"
        )
        val_dataset = dataset_mod.OriginalDataset(
            data_list=val_list,
            transform=self.data_transform,
            phase="val"
        )
        ## dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
        return dataloaders_dict

    def getNetwork(self, resize):
        net = network_mod.Network(resize, use_pretrained=True)
        print(net)
        net.to(self.device)
        return net

    def getOptimizer(self, optimizer_name, lr_cnn, lr_fc):
        ## param
        list_cnn_param_value, list_fc_param_value = self.net.getParamValueList()
        ## optimizer
        if optimizer_name == "SGD":
            optimizer = optim.SGD([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value, "lr": lr_fc}
            ], momentum=0.9)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value, "lr": lr_fc}
            ])
        print(optimizer)
        return optimizer

    def getStrHyperparameter(self, method_name, resize, mean_element, std_element, optimizer_name, lr_cnn, lr_fc, batch_size):
        str_hyperparameter = method_name \
            + str(len(self.dataloaders_dict["train"].dataset)) + "train" \
            + str(len(self.dataloaders_dict["val"].dataset)) + "val" \
            + str(resize) + "resize" \
            + str(mean_element) + "mean" \
            + str(std_element) + "std" \
            + optimizer_name \
            + str(lr_cnn) + "lrcnn" \
            + str(lr_fc) + "lrfc" \
            + str(batch_size) + "batch" \
            + str(self.num_epochs) + "epoch"
        print("str_hyperparameter = ", str_hyperparameter)
        return str_hyperparameter

    def train(self):
        ## time
        start_clock = time.time()
        ## loss record
        writer = SummaryWriter(logdir = "../../logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + self.str_hyperparameter)
        record_loss_train = []
        record_loss_val = []
        ## loop
        for epoch in range(self.num_epochs):
            print("----------")
            print("Epoch {}/{}".format(epoch+1, self.num_epochs))
            ## phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.net.train()
                else:
                    self.net.eval()
                ## skip
                if (epoch == 0) and (phase=="train"):
                    continue
                ## data load
                epoch_loss = 0.0
                for inputs, labels in tqdm(self.dataloaders_dict[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    ## reset gradient
                    self.optimizer.zero_grad()   #reset grad to zero (after .step())
                    ## compute gradient
                    with torch.set_grad_enabled(phase == "train"):  #compute grad only in "train"
                        ## forward
                        outputs = self.net(inputs)
                        loss = self.computeLoss(outputs, labels)
                        ## backward
                        if phase == "train":
                            loss.backward()     #accumulate gradient to each Tensor
                            self.optimizer.step()    #update param depending on current .grad
                        ## add loss
                        epoch_loss += loss.item() * inputs.size(0)
                ## average loss
                epoch_loss = epoch_loss / len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))
                ## record
                if phase == "train":
                    record_loss_train.append(epoch_loss)
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                else:
                    record_loss_val.append(epoch_loss)
                    writer.add_scalar("Loss/val", epoch_loss, epoch)
            if record_loss_train and record_loss_val:
                writer.add_scalars("Loss/train_and_val", {"train": record_loss_train[-1], "val": record_loss_val[-1]}, epoch)
        writer.close()
        ## save
        self.saveParam()
        self.saveGraph(record_loss_train, record_loss_val)
        ## training time
        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print ("training_time: ", mins, " [min] ", secs, " [sec]")

    def computeLoss(self, outputs, labels):
        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)
        return loss

    def saveParam(self):
        save_path = "../../weights/" + self.str_hyperparameter + ".pth"
        torch.save(self.net.state_dict(), save_path)
        print("Saved: ", save_path)

    def saveGraph(self, record_loss_train, record_loss_val):
        graph = plt.figure()
        plt.plot(range(len(record_loss_train)), record_loss_train, label="Training")
        plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss [m^2/s^4]")
        plt.title("loss: train=" + str(record_loss_train[-1]) + ", val=" + str(record_loss_val[-1]))
        graph.savefig("../../graph/" + self.str_hyperparameter + ".jpg")
        plt.show()

def main():
    ## hyperparameters
    method_name = "regression"
    train_rootpath = "../../../dataset_image_to_gravity/AirSim/1cam/train"
    val_rootpath = "../../../dataset_image_to_gravity/AirSim/1cam/val"
    csv_name = "imu_camera.csv"
    resize = 112
    mean_element = 0.5
    std_element = 0.5
    optimizer_name = "Adam"  #"SGD" or "Adam"
    lr_cnn = 1e-5
    lr_fc = 1e-4
    batch_size = 100
    num_epochs = 50
    ## train
    train_model = TrainModel(
        method_name,
        train_rootpath, val_rootpath, csv_name,
        resize, mean_element, std_element,
        optimizer_name, lr_cnn, lr_fc,
        batch_size, num_epochs
    )
    train_model.train()

if __name__ == '__main__':
    main()
