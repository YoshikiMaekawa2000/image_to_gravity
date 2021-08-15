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
        self.list_error_g_angle = []

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
            ## error in roll and pitch
            label_rp = self.accToRP(acc)
            error_rp = self.computeAngleDiff(ave_rp, label_rp)
            self.list_error_rp.append(error_rp)
            ## error in angle of g
            error_g_angle = self.getAngleBetweenVectors(acc, ave_acc)
            self.list_error_g_angle.append(error_g_angle)
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

    def getAngleBetweenVectors(v1, v2):
        return math.acos(np.dot(v1, v2)/np.linalg.norm(v1, ord=2)/np.linalg.norm(v2, ord=2))

    def printError(self):
        mae_rp = self.computeMAE(np.array(self.list_error_rp)/math.pi*180.0)
        var_rp = self.computeVar(np.array(self.list_error_rp)/math.pi*180.0)
        print("mae_rp [deg] = ", mae_rp)
        print("var_rp [deg^2] = ", var_rp)
        mae_g_angle = self.computeMAE(np.array(self.list_error_g_angle)/math.pi*180.0)
        var_g_angle = self.computeVar(np.array(self.list_error_g_angle)/math.pi*180.0)
        print("mae_g_angle [deg] = ", mae_g_angle)
        print("var_g_angle [deg^2] = ", var_g_angle)

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
