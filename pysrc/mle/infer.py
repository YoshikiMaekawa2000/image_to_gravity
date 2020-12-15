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
import criterion_mod

class Sample(inference_mod.Sample):
    def __init__(self,
            index,
            inputs_path, inputs, label, mu, cov, mul_sigma,
            label_r, label_p, output_r, output_p, error_r, error_p):
        super(Sample, self).__init__(
            index,
            inputs_path, inputs, label, mu,
            label_r, label_p, output_r, output_p, error_r, error_p
        )
        self.cov = cov              #ndarray
        self.mul_sigma = mul_sigma  #float

    def printData(self):
        super(Sample, self).printData()
        print("cov: ", self.cov)
        print("mul_sigma: ", self.mul_sigma)

class Inference(inference_mod.Inference):
    def __init__(self,
            dataset,
            net, weights_path, criterion,
            batch_size,
            th_mul_sigma):
        super(Inference, self).__init__(
            dataset,
            net, weights_path, criterion,
            batch_size
        )
        ## list
        self.list_selected_samples = []
        self.list_cov = []
        ## threshold
        self.th_mul_sigma = th_mul_sigma

    def infer(self):    #overwrite
        ## time
        start_clock = time.time()
        ## data load
        loss_all = 0.0
        for inputs, labels in tqdm(self.dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            ## compute gradient
            with torch.set_grad_enabled(False):
                ## forward
                outputs = self.net(inputs)
                loss_batch = self.computeLoss(outputs, labels)
                ## add loss
                loss_all += loss_batch.item() * inputs.size(0)
                # print("loss_batch.item() = ", loss_batch.item())
            ## append
            self.list_inputs += list(inputs.cpu().detach().numpy())
            self.list_labels += labels.cpu().detach().numpy().tolist()
            self.list_outputs += outputs.cpu().detach().numpy().tolist()
            cov = self.criterion.getCovMatrix(outputs)
            self.list_cov += list(cov.cpu().detach().numpy())
        ## compute error
        mae, var, ave_mul_sigma, selected_mae, selected_var = self.computeAttitudeError()
        ## sort
        self.sortSamples()
        ## show result & set graph
        self.showResult()
        print ("-----")
        ## inference time
        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print ("inference time: ", mins, " [min] ", secs, " [sec]")
        ## average loss
        loss_all = loss_all / len(self.dataloader.dataset)
        print("Loss: {:.4f}".format(loss_all))
        ## MAE & Var
        print("mae [deg] = ", mae)
        print("var [deg^2] = ", var)
        ## average multiplied sigma
        print("ave_mul_sigma [m^3/s^6] = ", ave_mul_sigma)
        ## selected MAE & Var
        print("th_mul_sigma = ", self.th_mul_sigma)
        print("number of the selected samples = ", len(self.list_selected_samples), " / ", len(self.list_samples))
        print("selected mae [deg] = ", selected_mae)
        print("selected var [deg^2] = ", selected_var)
        ## graph
        plt.tight_layout()
        plt.show()

    def computeAttitudeError(self): #overwrite
        list_errors = []
        list_selected_errors = []
        list_mul_sigma = []
        for i in range(len(self.list_labels)):
            ## error
            label_r, label_p = self.accToRP(self.list_labels[i])
            output_r, output_p = self.accToRP(self.list_outputs[i])
            error_r = self.computeAngleDiff(output_r, label_r)
            error_p = self.computeAngleDiff(output_p, label_p)
            list_errors.append([error_r, error_p])
            ## multiplied sigma
            mul_sigma = math.sqrt(self.list_cov[i][0, 0]) * math.sqrt(self.list_cov[i][1, 1]) * math.sqrt(self.list_cov[i][2, 2])
            list_mul_sigma.append(mul_sigma)
            ## register
            sample = Sample(
                i,
                self.dataloader.dataset.data_list[i][3:], self.list_inputs[i], self.list_labels[i], self.list_outputs[i], self.list_cov[i], mul_sigma,
                label_r, label_p, output_r, output_p, error_r, error_p
            )
            self.list_samples.append(sample)
            ## judge
            if mul_sigma < self.th_mul_sigma:
                self.list_selected_samples.append(sample)
                list_selected_errors.append([error_r, error_p])
        arr_errors = np.array(list_errors)
        arr_selected_errors = np.array(list_selected_errors)
        print("arr_errors.shape = ", arr_errors.shape)
        mae = self.computeMAE(arr_errors/math.pi*180.0)
        var = self.computeVar(arr_errors/math.pi*180.0)
        ave_mul_sigma = np.mean(list_mul_sigma, axis=0)
        selected_mae = self.computeMAE(arr_selected_errors/math.pi*180.0)
        selected_var = self.computeVar(arr_selected_errors/math.pi*180.0)
        return mae, var, ave_mul_sigma, selected_mae, selected_var

    def sortSamples(self):  #overwrite
        list_sum_error_rp = [abs(sample.error_r) + abs(sample.error_p) for sample in self.list_samples]
        list_mul_sigma = [sample.mul_sigma for sample in self.list_samples]
        ## get indicies
        sorted_indicies = np.argsort(list_sum_error_rp)         #error: small->large
        # sorted_indicies = np.argsort(list_sum_error_rp)[::-1]   #error: large->small
        # sorted_indicies = np.argsort(list_mul_sigma)            #sigma: small->large
        # sorted_indicies = np.argsort(list_mul_sigma)[::-1]      #sigma: large->small
        ## sort
        self.list_samples = [self.list_samples[index] for index in sorted_indicies]

def main():
    ## hyperparameters
    rootpath = "../../../dataset_image_to_gravity/AirSim/1cam/val"
    csv_name = "imu_camera.csv"
    resize = 112
    mean_element = 0.5
    std_element = 0.5
    batch_size = 10
    weights_path = "../../weights/mle.pth"
    th_mul_sigma = 0.0001
    ## dataset
    dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element])
        ),
        phase="val"
    )
    ## network
    net = network_mod.Network(resize, dim_fc_out=9, use_pretrained_vgg=False)
    ## criterion
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = criterion_mod.Criterion(device)
    ## infer
    inference = Inference(
        dataset,
        net, weights_path, criterion,
        batch_size,
        th_mul_sigma
    )
    inference.infer()

if __name__ == '__main__':
    main()
