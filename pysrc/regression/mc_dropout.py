import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import time

import torch
from torchvision import models
import torch.nn as nn

import sys
sys.path.append('../')
from common import inference_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import network_mod

class Sample(inference_mod.Sample):
    def __init__(self,
            index,
            inputs_path, inputs, label,
            mean, var, std_dist,
            label_r, label_p, output_r, output_p, error_r, error_p):
        super(Sample, self).__init__(
            index,
            inputs_path, inputs, label, mean,
            label_r, label_p, output_r, output_p, error_r, error_p
        )
        self.var = var              #list
        self.std_dist = std_dist    #float

    def printData(self):
        super(Sample, self).printData()
        print("var[m^2/s^4]: ", self.var)
        print("std_dist[m/s^2]: ", self.std_dist)

class Inference(inference_mod.Inference):
    def __init__(self,
            dataset,
            net, weights_path, criterion,
            batch_size,
            num_sampling, th_std_dist):
        super(Inference, self).__init__(
            dataset,
            net, weights_path, criterion,
            batch_size
        )
        ## parameters
        self.num_sampling = num_sampling
        self.th_std_dist = th_std_dist
        ## list
        self.list_selected_samples = []
        self.list_mean = []
        self.list_var = []
        self.list_std_dist = []
        ## set
        self.enable_dropout()

    def enable_dropout(self):
        for module in self.net.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

    def infer(self):    #overwrite
        ## time
        start_clock = time.time()
        ## data load
        loss_all = 0.0
        for inputs, labels in tqdm(self.dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.set_grad_enabled(False):
                list_outputs = []
                for _ in range(self.num_sampling):
                    ## forward
                    outputs = self.net(inputs)
                    loss_batch = self.computeLoss(outputs, labels)
                    ## add
                    list_outputs.append(outputs.unsqueeze(0))
                    loss_all += loss_batch.item() * inputs.size(0)
            outputs_mean = torch.cat(list_outputs, 0).mean(0)
            outputs_var = torch.cat(list_outputs, 0).var(0)
            ## append
            self.list_inputs += list(inputs.cpu().detach().numpy())
            self.list_labels += labels.cpu().detach().numpy().tolist()
            self.list_mean += outputs_mean.cpu().detach().numpy().tolist()
            self.list_var += outputs_var.cpu().detach().numpy().tolist()
        ## compute error
        mae, var, ave_std_dist, selected_mae, selected_var = self.computeAttitudeError()
        ## sort
        self.sortSamples()
        ## show result & set graph
        self.showResult()
        print ("-----")
        ## inference time
        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print ("inference time: ", mins, " [min] ", secs, " [sec]")
        ## result
        loss_all = loss_all / len(self.dataloader.dataset)
        print("Loss: {:.4f}".format(loss_all))
        print("mae [deg] = ", mae)
        print("var [deg^2] = ", var)
        print("ave_std_dist [m/s^2] = ", ave_std_dist)
        print("th_std_dist = ", self.th_std_dist)
        print("#selected samples = ", len(self.list_selected_samples), " / ", len(self.list_samples))
        print("selected mae [deg] = ", selected_mae)
        print("selected var [deg^2] = ", selected_var)
        ## graph
        plt.tight_layout()
        plt.show()

    def computeAttitudeError(self): #overwrite
        list_errors = []
        list_selected_errors = []
        for i in range(len(self.list_labels)):
            ## error
            label_r, label_p = self.accToRP(self.list_labels[i])
            output_r, output_p = self.accToRP(self.list_mean[i])
            error_r = self.computeAngleDiff(output_r, label_r)
            error_p = self.computeAngleDiff(output_p, label_p)
            list_errors.append([error_r, error_p])
            ## std distance
            std_dist = math.sqrt(self.list_var[i][0] + self.list_var[i][1] + self.list_var[i][2])
            self.list_std_dist.append(std_dist)
            print("std_dist = ", std_dist)
            ## register
            sample = Sample(
                i,
                self.dataloader.dataset.data_list[i][3:], self.list_inputs[i], self.list_labels[i],
                self.list_mean[i], self.list_var[i], std_dist,
                label_r, label_p, output_r, output_p, error_r, error_p
            )
            self.list_samples.append(sample)
            ## judge
            if std_dist < self.th_std_dist:
                self.list_selected_samples.append(sample)
                list_selected_errors.append([error_r, error_p])
        arr_errors = np.array(list_errors)
        arr_selected_errors = np.array(list_selected_errors)
        print("arr_errors.shape = ", arr_errors.shape)
        mae = self.computeMAE(arr_errors/math.pi*180.0)
        var = self.computeVar(arr_errors/math.pi*180.0)
        ave_std_dist = np.mean(self.list_std_dist, axis=0)
        selected_mae = self.computeMAE(arr_selected_errors/math.pi*180.0)
        selected_var = self.computeVar(arr_selected_errors/math.pi*180.0)
        return mae, var, ave_std_dist, selected_mae, selected_var

    def sortSamples(self):  #overwrite
        list_sum_error_rp = [abs(sample.error_r) + abs(sample.error_p) for sample in self.list_samples]
        ## get indicies
        # sorted_indicies = np.argsort(list_sum_error_rp)         #error: small->large
        # sorted_indicies = np.argsort(list_sum_error_rp)[::-1]   #error: large->small
        sorted_indicies = np.argsort(self.list_std_dist)        #sigma: small->large
        # sorted_indicies = np.argsort(self.list_std_dist)[::-1]  #sigma: large->small
        ## sort
        self.list_samples = [self.list_samples[index] for index in sorted_indicies]

def main():
    ## hyperparameters
    list_rootpath = ["../../../dataset_image_to_gravity/AirSim/1cam/val"]
    csv_name = "imu_camera.csv"
    resize = 224
    mean_element = 0.5
    std_element = 0.5
    batch_size = 10
    weights_path = "../../weights/regression.pth"
    num_sampling = 50
    th_std_dist = 0.05
    ## dataset
    dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(list_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element])
        ),
        phase="val"
    )
    ## network
    net = network_mod.Network(resize, dim_fc_out=3, use_pretrained_vgg=False)
    ## criterion
    criterion = nn.MSELoss()
    ## infer
    inference = Inference(
        dataset,
        net, weights_path, criterion,
        batch_size,
        num_sampling, th_std_dist
    )
    inference.infer()

if __name__ == '__main__':
    main()
