from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, resize, mean, std):
        self.img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img_pil, acc_numpy, phase="train"):
        ## augemntation
        if phase == "train":
            ## mirror
            is_mirror = bool(random.getrandbits(1))
            if is_mirror:
                ## image
                img_pil = ImageOps.mirror(img_pil)
                ## acc
                acc_numpy[1] = -acc_numpy[1]
            ## rotate
            angle_deg = random.uniform(-10.0, 10.0)
            angle_rad = angle_deg / 180 * math.pi
            ## image
            img_pil = img_pil.rotate(angle_deg)
            ## acc
            acc_numpy = self.rotateVector(acc_numpy, angle_rad)
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        ## acc: numpy -> tensor
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return img_tensor, acc_tensor

    def rotateVector(self, acc_numpy, angle):
        rot = np.array([
            [1, 0, 0],
            [0, math.cos(-angle), -math.sin(-angle)],
            [0, math.sin(-angle), math.cos(-angle)]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

##### test #####
# ## image
# img_path = "../../../dataset_image_to_gravity/AirSim/example/camera_0.jpg"
# img_pil = Image.open(img_path)
# ## label
# acc_list = [1, 0, 0]
# acc_numpy = np.array(acc_list)
# print("acc_numpy = ", acc_numpy)
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## transform
# transform = DataTransform(resize, mean, std)
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
