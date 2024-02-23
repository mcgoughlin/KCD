import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSimilarityLoss(nn.Module):
    def __init__(self):
        super(LatentSimilarityLoss, self).__init__()
        self.similarity_loss = nn.MSELoss()

    def normalize(self, z):
        mag = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        return z / mag

    def forward(self, z1, z2):
        b,d,x,y,z = z1.shape
        return self.similarity_loss(self.normalize(z1.view(b,d,-1)), self.normalize(z2.view(b,d,-1)))

class PyramidalLatentSimilarityLoss(nn.Module):
    def __init__(self):
        super(PyramidalLatentSimilarityLoss, self).__init__()
        self.similarity_loss = LatentSimilarityLoss()

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