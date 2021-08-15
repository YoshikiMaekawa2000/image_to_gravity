import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import time
from PIL import Image

import torch
from torchvision import models
import torch.nn as nn

class Sample:
    def __init__(self,
            index,
            inputs_path, inputs, label, mean,
            label_r, label_p, output_r, output_p, error_r, error_p):
        self.index = index              #int
        self.inputs_path = inputs_path  #list
        self.inputs = inputs            #ndarray
        self.label = label              #list
        self.mean = mean                #list
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
        print("mean: ", self.mean)
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
        self.list_est = []

    def getDataloader(self, dataset, batch_size):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
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
            self.list_est += outputs.cpu().detach().numpy().tolist()
        ## compute error
        mae_rp, var_rp, mae_g_angle, var_g_angle = self.computeAttitudeError()
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
        print("mae_rp [deg] = ", mae_rp)
        print("var_rp [deg^2] = ", var_rp)
        print("mae_g_angle [deg] = ", mae_g_angle)
        print("var_g_angle [deg^2] = ", var_g_angle)
        ## graph
        plt.tight_layout()
        plt.show()

    def computeLoss(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss

    def computeAttitudeError(self):
        list_errors_rp = []
        list_errors_g_angle = []
        for i in range(len(self.list_labels)):
            ## error
            label_r, label_p = self.accToRP(self.list_labels[i])
            output_r, output_p = self.accToRP(self.list_est[i])
            error_r = self.computeAngleDiff(output_r, label_r)
            error_p = self.computeAngleDiff(output_p, label_p)
            list_errors_rp.append([error_r, error_p])
            ## error
            error_g_angle = self.getAngleBetweenVectors(self.list_labels[i], self.list_est[i])
            list_errors_g_angle.append(error_g_angle)
            ## register
            sample = Sample(
                i,
                self.dataloader.dataset.data_list[i][3:], self.list_inputs[i], self.list_labels[i], self.list_est[i],
                label_r, label_p, output_r, output_p, error_r, error_p
            )
            self.list_samples.append(sample)
        mae_rp = self.computeMAE(np.array(list_errors_rp)/math.pi*180.0)
        var_rp = self.computeVar(np.array(list_errors_rp)/math.pi*180.0)
        mae_g_angle = self.computeMAE(np.array(list_errors_g_angle)/math.pi*180.0)
        var_g_angle = self.computeVar(np.array(list_errors_g_angle)/math.pi*180.0)
        return mae_rp, var_rp, mae_g_angle, var_g_angle

    def accToRP(self, acc):
        r = math.atan2(acc[1], acc[2])
        p = math.atan2(-acc[0], math.sqrt(acc[1]*acc[1] + acc[2]*acc[2]))
        return r, p

    def computeAngleDiff(self, angle1, angle2):
        diff = math.atan2(math.sin(angle1 - angle2), math.cos(angle1 - angle2))
        return diff

    def getAngleBetweenVectors(v1, v2):
        return math.acos(np.dot(v1, v2)/np.linalg.norm(v1, ord=2)/np.linalg.norm(v2, ord=2))

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
        visualize_gravity = True
        plt.figure()
        h = 2
        w = 5
        num_shown = h*w
        if visualize_gravity:
            h = 2*h
        for i, sample in enumerate(self.list_samples):
            sample.printData()
            if i < num_shown:
                ## gravity 
                if visualize_gravity:
                    i = i + w*(i//w)
                    fig = plt.subplot(h, w, i+1+w)
                    fig.set_xlim(-1, 1)
                    fig.set_ylim(-1, 1)
                    fig.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                    fig.quiver(-0.5, 0, color='green', angles='xy', scale_units='xy', scale=1)
                    fig.quiver(0, 0.5, color='blue', angles='xy', scale_units='xy', scale=1)
                    fig.quiver(sample.label[1], -sample.label[2], color='deepskyblue', angles='xy', scale_units='xy', scale=1)
                    fig.quiver(sample.mean[1], -sample.mean[2], color='magenta', angles='xy', scale_units='xy', scale=1)
                ## image
                plt.subplot(h, w, i+1)
                plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                plt.imshow(Image.open(sample.inputs_path[0]))
                # plt.imshow(np.clip(sample.inputs.transpose((1, 2, 0)), 0, 1))
                plt.title(str(sample.index))
