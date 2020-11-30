from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, resize, mean, std, hor_fov_deg=-1):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.hor_fov_rad = hor_fov_deg / 180.0 * math.pi

    def __call__(self, img_pil, acc_numpy, phase="train"):
        ## augemntation
        if phase == "train":
            ## mirror
            is_mirror = bool(random.getrandbits(1))
            if is_mirror:
                img_pil, acc_numpy = self.mirror(img_pil, acc_numpy)
            ## homography
            if 0 < self.hor_fov_rad < math.pi:
                img_pil, acc_numpy = self.randomHomography(img_pil, acc_numpy)
            ## rotation
            img_pil, acc_numpy = self.randomRotation(img_pil, acc_numpy)
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        ## acc: numpy -> tensor
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return img_tensor, acc_tensor

    def mirror(self, img_pil, acc_numpy):
        ## image
        img_pil = ImageOps.mirror(img_pil)
        ## acc
        acc_numpy[1] = -acc_numpy[1]
        return img_pil, acc_numpy

    def randomHomography(self, img_pil, acc_numpy):
        angle_rad = random.uniform(-10.0, 10.0) / 180.0 * math.pi
        # print("hom: angle_rad/math.pi*180.0 = ", angle_rad/math.pi*180.0)
        ## image
        (w, h) = img_pil.size
        ver_fov_rad = h / w * self.hor_fov_rad
        ## tilt
        d = abs(h * math.sin(angle_rad))
        large_h = h + 2 * d * math.tan(ver_fov_rad / 2)
        small_h = h * math.cos(angle_rad) + d * math.tan(ver_fov_rad / 2)
        large_w = w + 2 * d * math.tan(self.hor_fov_rad / 2)
        small_w = w
        ## scalling
        shrunk_h = h * small_h / large_h
        shrunk_w = w * small_w / large_w
        ## transform
        if angle_rad > 0:
            points_after = [(0, 0), (w, 0), ((w + shrunk_w)//2, shrunk_h), ((w - shrunk_w)//2, shrunk_h)]
        else:
            points_after = [((w - shrunk_w)//2, h - shrunk_h), ((w + shrunk_w)//2, h - shrunk_h), (w, h), (0, h)]
        points_before = [(0, 0), (w, 0), (w, h), (0, h)]
        coeffs = self.find_coeffs(points_after, points_before)
        img_pil = img_pil.transform(img_pil.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        ## acc
        acc_numpy = self.rotateVectorPitch(acc_numpy, -angle_rad)
        return img_pil, acc_numpy

    ## copied from "http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil"
    def find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        ret = np.array(res).reshape(8)
        return ret

    def rotateVectorPitch(self, acc_numpy, angle):
        rot = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

    def randomRotation(self, img_pil, acc_numpy):
        angle_deg = random.uniform(-10.0, 10.0)
        angle_rad = angle_deg / 180 * math.pi
        # print("rot: angle_deg = ", angle_deg)
        ## image
        img_pil = img_pil.rotate(angle_deg)
        ## acc
        acc_numpy = self.rotateVectorRoll(acc_numpy, -angle_rad)
        return img_pil, acc_numpy

    def rotateVectorRoll(self, acc_numpy, angle):
        rot = np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

##### test #####
# ## image
# img_path = "../../../dataset_image_to_gravity/AirSim/example/camera_0.jpg"
# img_pil = Image.open(img_path)
# ## label
# acc_list = [0, 0, 1]
# acc_numpy = np.array(acc_list)
# print("acc_numpy = ", acc_numpy)
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# hor_fov_deg = 70
# ## transform
# transform = DataTransform(resize, mean, std, hor_fov_deg=hor_fov_deg)
# img_trans, acc_trans = transform(img_pil, acc_numpy, phase="train")
# print("acc_trans = ", acc_trans)
# ## tensor -> numpy
# img_trans_numpy = img_trans.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
# img_trans_numpy = np.clip(img_trans_numpy, 0, 1)
# print("img_trans_numpy.shape = ", img_trans_numpy.shape)
# ## save
# img_trans_pil = Image.fromarray(np.uint8(255*img_trans_numpy))
# save_path = "../../save/transform.jpg"
# img_trans_pil.save(save_path)
# print("saved: ", save_path)
# ## imshow
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.imshow(img_pil)
# plt.subplot(2, 1, 2)
# plt.imshow(img_trans_numpy)
# plt.show()
