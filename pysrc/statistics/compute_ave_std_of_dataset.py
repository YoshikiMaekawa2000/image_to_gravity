import numpy as np
import math

import sys
sys.path.append('../')

from common import make_datalist_mod

class StatisticsModel:
    def __init__(self, list_rootpath, csv_name):
        ## list
        self.list_data = make_datalist_mod.makeDataList(list_rootpath, csv_name)
        self.list_error_rp = []

    def __call__(self):
        self.computeAttitudeError()
        self.printError()

    def computeAttitudeError(self):
        ## average
        list_acc = [data[:3] for data in self.list_data]
        ave_acc = np.mean(np.array(list_acc).astype(float), axis=0)
        ave_rp = self.accToRP(ave_acc)
        ## error
        for acc in list_acc:
            acc = [float(num) for num in acc]
            label_rp = self.accToRP(acc)
            error_rp = self.computeAngleDiff(ave_rp, label_rp)
            self.list_error_rp.append(error_rp)
        ## print
        print("ave_rp [deg] = ", ave_rp/math.pi*180.0)

    def accToRP(self, acc):
        r = math.atan2(acc[1], acc[2])
        p = math.atan2(-acc[0], math.sqrt(acc[1]*acc[1] + acc[2]*acc[2]))
        rp = np.array([r, p])
        return rp

    def computeAngleDiff(self, angle1, angle2):
        diff = np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))
        return diff

    def printError(self):
        mae = self.computeMAE(np.array(self.list_error_rp)/math.pi*180.0)
        var = self.computeVar(np.array(self.list_error_rp)/math.pi*180.0)
        print("mae [deg] = ", mae)
        print("var [deg^2] = ", var)

    def computeMAE(self, x):
        return np.mean(np.abs(x), axis=0)

    def computeVar(self, x):
        return np.var(x, axis=0)

def main():
    ## hyperparameters
    list_rootpath = ["../../../dataset_image_to_gravity/AirSim/lidar1cam/val"]
    csv_name = "imu_lidar_camera.csv"
    ## procrss
    statistics_model = StatisticsModel(list_rootpath, csv_name)
    statistics_model()

if __name__ == '__main__':
    main()
