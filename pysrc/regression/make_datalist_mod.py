import csv
import os

def makeDataList(rootpath, csv_name):
    csv_path = os.path.join(rootpath, csv_name)
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        data_list = []
        for row in reader:
            row[3] = os.path.join(rootpath, row[3])
            data_list.append(row)
    return data_list

##### test #####
# rootpath = "../../../dataset_image_to_gravity/AirSim/1cam/train"
# csv_name = "imu_camera.csv"
# train_list = makeDataList(rootpath, csv_name)
# # print(train_list)
# print("example0: ", train_list[0][:3], train_list[0][3:])
# print("example1: ", train_list[1][:3], train_list[1][3:])
