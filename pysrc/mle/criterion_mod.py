import numpy as np
import math

import torch

class Criterion:
    def __init__(self, device):
        self.device = device

    def __call__(self, outputs, labels):
        mu = outputs[:, :3]
        L = self.getTriangularMatrix(outputs)
        L = L.to(self.device)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=L)
        loss = -dist.log_prob(labels)
        loss = loss.mean()
        return loss

    def getTriangularMatrix(self, outputs):
        elements = outputs[:, 3:9]
        L = torch.zeros(outputs.size(0), elements.size(1)//2, elements.size(1)//2)
        L[:, 0, 0] = torch.exp(elements[:, 0])
        L[:, 1, 0] = elements[:, 1]
        L[:, 1, 1] = torch.exp(elements[:, 2])
        L[:, 2, 0] = elements[:, 3]
        L[:, 2, 1] = elements[:, 4]
        L[:, 2, 2] = torch.exp(elements[:, 5])
        return L

    def getCovMatrix(self, outputs):
        L = self.getTriangularMatrix(outputs)
        Ltrans = torch.transpose(L, 1, 2)
        LL = torch.bmm(L, Ltrans)
        return LL

#### test #####
# ## device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device = ", device)
# ## outputs
# outputs = np.array([
#     [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.5, 0.5, 0.5],
#     [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.5, 0.5, 0.5]
# ]).astype(np.float32)
# outputs = torch.from_numpy(outputs)
# outputs = outputs.to(device)
# print("outputs.size() = ", outputs.size())
# print("outputs = ", outputs)
# ## labels
# labels = np.array([
#     [2.1, 3.2, 4.3],
#     [2.1, 3.2, 4.3]
# ]).astype(np.float32)
# labels = torch.from_numpy(labels)
# labels = labels.to(device)
# print("labels.size() = ", labels.size())
# print("labels = ", labels)
# ## loss
# criterion = CriterionModel(device)
# loss = criterion(outputs, labels)
# print("loss.size() = ", loss.size())
# print("loss = ", loss)
