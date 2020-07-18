import numpy as np
import math

import torch

def getCoef(x):
    g = x[:, :3]
    s = x[:, 3:6]
    c = x[:, 6:9]
    return g, s, c

def getCov(x):
    _, s, c = getCoef(x)
    sxx = s[:, 0] * s[:, 0]
    syy = s[:, 1] * s[:, 1]
    szz = s[:, 2] * s[:, 2]
    sxy = c[:, 0] * s[:, 0] * s[:, 1]
    syz = c[:, 1] * s[:, 1] * s[:, 2]
    szx = c[:, 2] * s[:, 2] * s[:, 0]

    cov = torch.empty(s.size(0), 3, 3)
    cov[:, 0, 0] = sxx
    cov[:, 0, 1] = sxy
    cov[:, 0, 2] = szx
    cov[:, 1, 0] = sxy
    cov[:, 1, 1] = syy
    cov[:, 1, 2] = szx
    cov[:, 2, 0] = szx
    cov[:, 2, 1] = syz
    cov[:, 2, 2] = szz
    print("cov.size() = ", cov.size())
    print("cov = ", cov)
    return cov

def originalCriterion(outputs, labels):
    k = labels.size(1)
    print("k = ", k)
    cov = getCov(outputs)
    cov = cov.det()
    print("cov = ", cov)
    denominator = torch.sqrt(((2*math.pi)**k) * torch.det(cov))
    loss = torch.mean((outputs[:, :3] - labels)**2)
    g, s, c = getCoef(outputs)
    getCov(outputs)

    return loss

##### test #####
outputs = np.array([
    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.5, 0.5, 0.5],
    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.5, 0.5, 0.5]
])
outputs = torch.from_numpy(outputs)
print("outputs.size() = ", outputs.size())
print("outputs = ", outputs)
labels = np.array([
    [2.1, 3.2, 4.3],
    [2.1, 3.2, 4.3]
])
labels = torch.from_numpy(labels)
print("labels.size() = ", labels.size())
print("labels = ", labels)

loss = originalCriterion(outputs, labels)
print("loss = ", loss)
