import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSimilarityLoss(nn.Module):
    def __init__(self):
        super(LatentSimilarityLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.cce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = 0.07
        self.mse_weight = 0.2

    def normalize(self, z):
        mag = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        return z / mag

    def forward(self, z1, z2):
        b,d,x,y,z = z1.shape

        # sharpened_z1 = self.softmax(z1.view(b,d,-1).swapaxes(1,2)/self.temperature)
        # sharpened_z2 = self.softmax(z2.view(b,d,-1).swapaxes(1,2)/self.temperature)
        #
        # symmetric_cce = self.cce(sharpened_z1, sharpened_z2) + self.cce(sharpened_z2, sharpened_z1)
        if d != 2:
            mse = self.mse(z1, z2).mean()
            cos = self.cos(z1.view(b,d,-1), z2.view(b,d,-1)).mean()
            return mse + (1 - cos)
        else:
            # this is a 2D latent space, so we can use the cross entropy loss
            argmax_z1 = torch.argmax(z1, dim=1)
            argmax_z2 = torch.argmax(z2, dim=1)
            cce = self.cce(z1, argmax_z2) + self.cce(z2, argmax_z1)
            return cce



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