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
            mean, cov, mul_std,
            label_r, label_p, output_r, output_p, error_r, error_p):
        super(Sample, self).__init__(
            index,
            inputs_path, inputs, label, mean,
            label_r, label_p, output_r, output_p, error_r, error_p
        )
        self.cov = cov          #ndarray
        self.mul_std = mul_std  #float

    def printData(self):
        super(Sample, self).printData()
        print("cov[m^2/s^4]: \n", self.cov)
        print("mul_std[m^3/s^6]: ", self.mul_std)

class Inference(inference_mod.Inference):
    def __init__(self,
            dataset,
            net, weights_path, criterion,
            batch_size,
            num_mcsampling, th_mul_std):
        super(Inference, self).__init__(
            dataset,
            net, weights_path, criterion,
            batch_size
        )
        ## parameters
        self.num_mcsampling = num_mcsampling
        self.th_mul_std = th_mul_std
        ## list
        self.list_selected_samples = []
        self.list_cov = []
        self.list_mul_std = []
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
            list_outputs = []
            with torch.set_grad_enabled(False):
                for _ in range(self.num_mcsampling):
                    ## forward
                    outputs = self.net(inputs)
                    loss_batch = self.computeLoss(outputs, labels)
                    ## add
                    list_outputs.append(outputs.cpu().detach().numpy())
                    loss_all += loss_batch.item() * inputs.size(0)
            ## append
            self.list_inputs += list(inputs.cpu().detach().numpy())
            self.list_labels += labels.cpu().detach().numpy().tolist()
            self.list_est += np.array(list_outputs).mean(0).tolist()
            for outputs in list(np.array(list_outputs).transpose(1, 0, 2)):
                self.list_cov.append(np.cov(outputs, rowvar=False, bias=True))
        ## compute error
        mae, var, ave_mul_std, selected_mae, selected_var, weighted_mae = self.computeAttitudeError()
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
        loss_all = loss_all / len(self.dataloader.dataset) / self.num_mcsampling
        print("Loss: {:.4f}".format(loss_all))
        print("mae [deg] = ", mae)
        print("var [deg^2] = ", var)
        print("ave_mul_std [m^3/s^6] = ", ave_mul_std)
        print("th_mul_std = ", self.th_mul_std)
        print("#selected samples = ", len(self.list_selected_samples), " / ", len(self.list_samples))
        print("selected mae [deg] = ", selected_mae)
        print("selected var [deg^2] = ", selected_var)
        print("weighted mae [deg] = ", weighted_mae)
        ## graph
        plt.tight_layout()
        plt.show()

    def computeAttitudeError(self): #overwrite
        list_errors = []
        list_selected_errors = []
        for i in range(len(self.list_labels)):
            ## error
            label_r, label_p = self.accToRP(self.list_labels[i])
            output_r, output_p = self.accToRP(self.list_est[i])
            error_r = self.computeAngleDiff(output_r, label_r)
            error_p = self.computeAngleDiff(output_p, label_p)
            list_errors.append([error_r, error_p])
            ## std distance
            mul_std = math.sqrt(self.list_cov[i][0, 0]) * math.sqrt(self.list_cov[i][1, 1]) * math.sqrt(self.list_cov[i][2, 2])
            # std_dist = math.sqrt(self.list_cov[i][0, 0] + self.list_cov[i][1, 1] + self.list_cov[i][2, 2])
            self.list_mul_std.append(mul_std)
            ## register
            sample = Sample(
                i,
                self.dataloader.dataset.data_list[i][3:], self.list_inputs[i], self.list_labels[i],
                self.list_est[i], self.list_cov[i], mul_std,
                label_r, label_p, output_r, output_p, error_r, error_p
            )
            self.list_samples.append(sample)
            ## judge
            if mul_std < self.th_mul_std:
                self.list_selected_samples.append(sample)
                list_selected_errors.append([error_r, error_p])
        mae = self.computeMAE(np.array(list_errors)/math.pi*180.0)
        var = self.computeVar(np.array(list_errors)/math.pi*180.0)
        ave_mul_std = np.mean(self.list_mul_std, axis=0)
        selected_mae = self.computeMAE(np.array(list_selected_errors)/math.pi*180.0)
        selected_var = self.computeVar(np.array(list_selected_errors)/math.pi*180.0)
        list_weighted_error = list(np.array(list_errors)/math.pi*180.0 * (1/np.array(self.list_mul_std)[:, np.newaxis]))
        weighted_mae = np.sum(np.abs(list_weighted_error), axis=0) / np.sum(1/np.array(self.list_mul_std))
        return mae, var, ave_mul_std, selected_mae, selected_var, weighted_mae

    def sortSamples(self):  #overwrite
        list_sum_error_rp = [abs(sample.error_r) + abs(sample.error_p) for sample in self.list_samples]
        ## get indicies
        # sorted_indicies = np.argsort(list_sum_error_rp)         #error: small->large
        # sorted_indicies = np.argsort(list_sum_error_rp)[::-1]   #error: large->small
        sorted_indicies = np.argsort(self.list_mul_std)        #sigma: small->large
        # sorted_indicies = np.argsort(self.list_mul_std)[::-1]  #sigma: large->small
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
    num_mcsampling = 50
    th_mul_std = 0.05
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
        num_mcsampling, th_mul_std
    )
    inference.infer()

if __name__ == '__main__':
    main()
