import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np


class LatentSimilarityLoss(nn.Module):
    def __init__(self, return_l2_latent=False, return_l1_latent=False,
                 return_symmetric_cce=False):
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
        self.return_l2_latent = return_l2_latent
        self.return_l1_latent = return_l1_latent
        self.return_symmetric_cce = return_symmetric_cce

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

        if d == 4:
            if self.return_symmetric_cce:
                amax_z1 = torch.argmax(reshaped_z1, dim=1)
                amax_z2 = torch.argmax(reshaped_z2, dim=1)
                symm_cce = (self.cce(reshaped_z1, amax_z2) + self.cce(reshaped_z2, amax_z1)) / 2
                return symm_cce #* ((x**3)/(64*64*64)) # scales down loss with depth (cubed)
            else:
                return 0
        else:
            if self.return_l2_latent:
                return self.l2(reshaped_z1, reshaped_z2) * (1e3/(d**3))
            elif self.return_l1_latent:
                return self.huber(reshaped_z1, reshaped_z2) * (1e3/(d**3))
            else:
                return 0




class PyramidalLatentSimilarityLoss(nn.Module):
    def __init__(self, return_l2_latent=False, return_l1_latent=False, return_symmetric_cce=True):
        super(PyramidalLatentSimilarityLoss, self).__init__()
        self.similarity_loss = LatentSimilarityLoss(return_l2_latent=return_l2_latent, return_l1_latent=return_l1_latent,
                                                    return_symmetric_cce=return_symmetric_cce)

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