import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import time

import torch
from torchvision import models
import torch.nn as nn

class Sample:
    def __init__(self,
            index,
            inputs_path, inputs, label, mu,
            label_r, label_p, output_r, output_p, error_r, error_p):
        self.index = index              #int
        self.inputs_path = inputs_path  #list
        self.inputs = inputs            #ndarray
        self.label = label              #list
        self.mu = mu                    #list
        self.label_r = label_r          #float
        self.label_p = label_p          #float
        self.output_r = output_r        #float
        self.output_p = output_p        #float
        self.error_r = error_r          #float
        self.error_p = error_p          #float

    def printData(self):
        print("-----", self.index, "-----")
        print("inputs_path: ", self.inputs_path)
        # print("inputs: ", self.inputs)
        print("inputs.shape: ", self.inputs.shape)
        print("label: ", self.label)
        print("mu: ", self.mu)
        print("l_r[deg]: ", self.label_r/math.pi*180.0, ", l_p[deg]: ", self.label_p/math.pi*180.0)
        print("o_r[deg]: ", self.output_r/math.pi*180.0, ", o_p[deg]: ", self.output_p/math.pi*180.0)
        print("e_r[deg]: ", self.error_r/math.pi*180.0, ", e_p[deg]: ", self.error_p/math.pi*180.0)

class Inference:
    def __init__(self,
            dataset,
            net, weights_path, criterion,
            batch_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device = ", self.device)
        self.dataloader = self.getDataloader(dataset, batch_size)
        self.net = self.getSetNetwork(net, weights_path)
        self.criterion = criterion
        ## list
        self.list_samples = []
        self.list_inputs = []
        self.list_labels = []
        self.list_outputs = []

    def getDataloader(self, dataset, batch_size):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return dataloader

    def getSetNetwork(self, net, weights_path):
        print(net)
        net.to(self.device)
        net.eval()
        ## load
        if torch.cuda.is_available():
            loaded_weights = torch.load(weights_path)
            print("Loaded [GPU -> GPU]: ", weights_path)
        else:
            loaded_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("Loaded [GPU -> CPU]: ", weights_path)
        net.load_state_dict(loaded_weights)
        return net

    def infer(self):
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
        ## compute error
        mae, var = self.computeAttitudeError()
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
        ## graph
        plt.tight_layout()
        plt.show()

    def computeLoss(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss

    def computeAttitudeError(self):
        list_errors = []
        for i in range(len(self.list_labels)):
            ## error
            label_r, label_p = self.accToRP(self.list_labels[i])
            output_r, output_p = self.accToRP(self.list_outputs[i])
            error_r = self.computeAngleDiff(output_r, label_r)
            error_p = self.computeAngleDiff(output_p, label_p)
            list_errors.append([error_r, error_p])
            ## register
            sample = Sample(
                i,
                self.dataloader.dataset.data_list[i][3:], self.list_inputs[i], self.list_labels[i], self.list_outputs[i],
                label_r, label_p, output_r, output_p, error_r, error_p
            )
            self.list_samples.append(sample)
        arr_errors = np.array(list_errors)
        print("arr_errors.shape = ", arr_errors.shape)
        mae = self.computeMAE(arr_errors/math.pi*180.0)
        var = self.computeVar(arr_errors/math.pi*180.0)
        return mae, var

    def accToRP(self, acc):
        r = math.atan2(acc[1], acc[2])
        p = math.atan2(-acc[0], math.sqrt(acc[1]*acc[1] + acc[2]*acc[2]))
        return r, p

    def computeAngleDiff(self, angle1, angle2):
        diff = math.atan2(math.sin(angle1 - angle2), math.cos(angle1 - angle2))
        return diff

    def computeMAE(self, x):
        return np.mean(np.abs(x), axis=0)

    def computeVar(self, x):
        return np.var(x, axis=0)

    def sortSamples(self):
        list_sum_error_rp = [abs(sample.error_r) + abs(sample.error_p) for sample in self.list_samples]
        ## get indicies
        sorted_indicies = np.argsort(list_sum_error_rp)         #error: small->large
        # sorted_indicies = np.argsort(list_sum_error_rp)[::-1]   #error: large->small
        ## sort
        self.list_samples = [self.list_samples[index] for index in sorted_indicies]

    def showResult(self):
        plt.figure()
        h = 5
        w = 5
        for i in range(len(self.list_samples)):
            self.list_samples[i].printData()
            if i < h*w:
                plt.subplot(h, w, i+1)
                plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                plt.imshow(self.list_samples[i].inputs.transpose((1, 2, 0)))
                plt.title(str(self.list_samples[i].index))
