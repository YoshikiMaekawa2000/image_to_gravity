import numpy as np
import math

import torch

def getCov(x):
    sigma = x[:, 3:6].clone()
    corr = x[:, 6:9].clone()
    sxx = sigma[:, 0] * sigma[:, 0]
    syy = sigma[:, 1] * sigma[:, 1]
    szz = sigma[:, 2] * sigma[:, 2]
    sxy = corr[:, 0] * sigma[:, 0] * sigma[:, 1]
    syz = corr[:, 1] * sigma[:, 1] * sigma[:, 2]
    szx = corr[:, 2] * sigma[:, 2] * sigma[:, 0]

    cov = torch.empty(sigma.size(0), sigma.size(1), sigma.size(1))
    cov[:, 0, 0] = sxx
    cov[:, 0, 1] = sxy
    cov[:, 0, 2] = szx
    cov[:, 1, 0] = sxy
    cov[:, 1, 1] = syy
    cov[:, 1, 2] = syz
    cov[:, 2, 0] = szx
    cov[:, 2, 1] = syz
    cov[:, 2, 2] = szz
    return cov

def getTriangularMatrix(outputs):
    elements = outputs[:, 3:9].clone()
    L = torch.zeros(outputs.size(0), elements.size(1)//2, elements.size(1)//2)
    # L[:, 0, 0] = elements[:, 0]
    L[:, 0, 0] = torch.exp(elements[:, 0])
    L[:, 1, 0] = elements[:, 1]
    # L[:, 1, 1] = elements[:, 2]
    L[:, 1, 1] = torch.exp(elements[:, 2])
    L[:, 2, 0] = elements[:, 3]
    L[:, 2, 1] = elements[:, 4]
    # L[:, 2, 2] = elements[:, 5]
    L[:, 2, 2] = torch.exp(elements[:, 5])
    return L

def computeDet(m):
    det = m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1]) - (m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])) + (m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0]))
    return det

def originalCriterion(outputs, labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k = labels.size(1)
    mu = outputs[:, :3].clone()
    cov = getCov(outputs)
    L = getTriangularMatrix(outputs)
    Ltrans = torch.transpose(L, 1, 2)

    # cov_inv = torch.inverse(cov)
    # cov_det = computeDet(cov)
    # diff = labels - mu
    # diff = diff.unsqueeze_(1)
    # diff_trans = torch.transpose(diff, 1, 2)
    #
    # cov_inv = cov_inv.to(device)
    # numerator = torch.exp(-1 / 2 * torch.bmm(torch.bmm(diff, cov_inv), diff_trans))
    # numerator = numerator.clone().squeeze_()
    # # denominator = torch.sqrt(((2*math.pi)**k) * torch.det(cov)) #det() works in torch>1.2.0
    # denominator = torch.sqrt(((2*math.pi)**k) * cov_det)
    # denominator = denominator.to(device)
    # loss = numerator / denominator
    # epsiron = 1e-20
    # loss = -torch.log(torch.clamp(loss, min=epsiron))
    # print("loss = ", loss)
    # loss = torch.sum(loss) / loss.size(0)
    # print("loss = ", loss)

    cov = cov.to(device)
    L = L.to(device)
    Ltrans = Ltrans.to(device)
    LL = torch.bmm(L, Ltrans)
    # print("cov = ", cov)
    # dist = torch.distributions.MultivariateNormal(mu, cov)
    # dist = torch.distributions.MultivariateNormal(mu, LL)
    dist = torch.distributions.MultivariateNormal(mu, scale_tril=L)
    loss = -dist.log_prob(labels)
    print("loss = ", loss)
    for i in range(loss.size(0)):
        if torch.isnan(loss[i]):
            print("torch.isnan(loss[i]) = ", torch.isnan(loss[i]))
            print("outputs[i] = ", outputs[i])
            # print("cov[i] = ", cov[i])
            print("L[i] = ", L[i])
            print("Ltrans[i] = ", Ltrans[i])
            print("LL[i] = ", LL[i])
    loss = loss.mean()
    print("loss = ", loss)

    # loss = torch.mean((mu - labels)**2)
    return loss

##### test #####
# outputs = np.array([
#     [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.5, 0.5, 0.5],
#     [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.5, 0.5, 0.5]
# ]).astype(np.float32)
# outputs = torch.from_numpy(outputs)
# print("outputs.size() = ", outputs.size())
# print("outputs = ", outputs)
# labels = np.array([
#     [2.1, 3.2, 4.3],
#     [2.1, 3.2, 4.3]
# ]).astype(np.float32)
# labels = torch.from_numpy(labels)
# print("labels.size() = ", labels.size())
# print("labels = ", labels)
#
# loss = originalCriterion(outputs, labels)
# print("loss.size() = ", loss.size())
# print("loss = ", loss)
