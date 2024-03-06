import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np


class LatentSimilarityLoss(nn.Module):
    def __init__(self, l2_weight = 1, l1_weight = 1, cce_weight = 1, cos_weight = 1):
        super(LatentSimilarityLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.cce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.temperature = np.exp(0.07)
        self.mse_weight = 0.01
        self.cos_weight = 1
        self.huber = nn.SmoothL1Loss()
        self.l2 = nn.MSELoss()
        self.kld = nn.KLDivLoss()
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.cce_weight = cce_weight
        self.cos_weight = cos_weight

    def normalize(self, z):
        mag = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        return z / mag

    def forward(self, z1, z2):
        b,d,x,y,z = z1.shape

        # reshaped_z1 = z1.view(b,d,-1).swapaxes(1,2)
        # reshaped_z2 = z2.view(b,d,-1).swapaxes(1,2)
        #
        # s_ptd = self.softmax(reshaped_z1)
        # t_ptd = self.softmax(reshaped_z2)
        # self.symmetric_cce = (self.cce(s_ptd, t_ptd) + self.cce(t_ptd, s_ptd))/2
        #
        # return self.symmetric_cce/(x*y*z)

        reshaped_z1 = z1.view(b, d, -1)
        reshaped_z2 = z2.view(b, d, -1)

        loss = 0

        if d == 4:
            if self.cce_weight>0:
                amax_z1 = torch.argmax(reshaped_z1, dim=1)
                amax_z2 = torch.argmax(reshaped_z2, dim=1)
                symm_cce = (self.cce(reshaped_z1, amax_z2) + self.cce(reshaped_z2, amax_z1)) / 2
                loss += symm_cce*self.cce_weight
        else:
            if self.l2_weight>0:
                loss += self.l2(reshaped_z1, reshaped_z2) * (1e3/(d**3))*self.l2_weight
            if self.l1_weight>0:
                loss += self.huber(reshaped_z1, reshaped_z2) * (1e3/(d**3))*self.l1_weight
            if self.cos_weight>0:
                loss += (1 - self.cos(self.normalize(reshaped_z1), self.normalize(reshaped_z2)))*self.cos_weight

        return loss




class PyramidalLatentSimilarityLoss(nn.Module):
    def __init__(self, l2_weight = 0, l1_weight = 0, cce_weight = 0, cos_weight = 0):
        super(PyramidalLatentSimilarityLoss, self).__init__()
        self.similarity_loss = LatentSimilarityLoss(
            l2_weight = l2_weight,
            l1_weight = l1_weight,
            cce_weight = cce_weight,
            cos_weight = cos_weight
        )

    def forward(self, feature_list1, feature_list2):
        loss = 0
        for i in range(len(feature_list1)):
            loss += self.similarity_loss(feature_list1[i], feature_list2[i])
        return loss



if __name__ == "__main__":
    # Test the loss function
    loss = PyramidalLatentSimilarityLoss()

    test_highsim = True

    z1a = torch.randn(4,10)
    z2a = torch.randn(4,10)

    z1b = torch.randn(5,100)
    z2b = torch.randn(5,100)

    z1c = torch.randn(6,1000)
    z2c = torch.randn(6,1000)

    if test_highsim:
        z2a+= z1a*2
        z2b+= z1b*2
        z2c+= z1c*2

    print(loss([z1a,z1b,z1c],[z2a,z2b,z2c]))