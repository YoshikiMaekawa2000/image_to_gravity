import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, list_dim_fc_out=[100, 18, 3], dropout_rate=0.1, use_pretrained_vgg=True):
        super(Network, self).__init__()

        vgg = models.vgg16(pretrained=use_pretrained_vgg)
        self.cnn = vgg.features

        dim_fc_in = 512*(resize//32)*(resize//32)
        list_dim_fc_in = [dim_fc_in] + list_dim_fc_out
        list_fc = []
        for i in range(len(list_dim_fc_in) - 1):
            list_fc.append(nn.Linear(list_dim_fc_in[i], list_dim_fc_in[i+1]))
            if i != len(list_dim_fc_in) - 2:
                list_fc.append(nn.ReLU(inplace=True))
                list_fc.append(nn.Dropout(p=dropout_rate))
        self.fc = nn.Sequential(*list_fc)
        # self.initializeWeights()

    def initializeWeights(self):
        for m in self.fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def getParamValueList(self):
        list_cnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "cnn" in param_name:
                # print("cnn: ", param_name)
                list_cnn_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_cnn_param_value: ",list_cnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_cnn_param_value, list_fc_param_value

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        l2norm = torch.norm(x[:, :3].clone(), p=2, dim=1, keepdim=True)
        x[:, :3] = torch.div(x[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return x

##### test #####
# from PIL import Image
# import numpy as np
# import data_transform_mod
# ## image
# img_path = "../../../dataset_image_to_gravity/AirSim/example/camera_0.jpg"
# img_pil = Image.open(img_path)
# ## label
# acc_list = [0, 0, 1]
# acc_numpy = np.array(acc_list)
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## transform
# transform = data_transform_mod.DataTransform(resize, mean, std)
# img_trans, _ = transform(img_pil, acc_numpy, phase="train")
# ## network
# net = Network(resize, dim_fc_out=3, use_pretrained_vgg=True)
# print(net)
# list_cnn_param_value, list_fc_param_value = net.getParamValueList()
# ## prediction
# inputs = img_trans.unsqueeze_(0)
# print("inputs.size() = ", inputs.size())
# outputs = net(inputs)
# print("outputs.size() = ", outputs.size())
# print("outputs = ", outputs)
