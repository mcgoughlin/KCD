import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
from random import random
from math import ceil


# want to select random 128x128x128 patches from the 3D images, padding if need be,
# and then apply random rotations, flips, and noise to the patches, cropping them back to 64x64x64
# before returning them
class CrossPhaseDataset(Dataset):
    def __init__(self, ne_path, ce_path, device=None, patch_size=64,
                 blur_kernel=torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                 is_train=True,patches_per_case=10):
        assert (os.path.exists(ne_path)) and os.path.exists(ce_path)
        self.ne_path = ne_path
        self.ce_path = ce_path
        self.device = device

        #save cases as a list of cases present in both ne_path and ce_path
        self.cases = [case for case in os.listdir(ne_path) if case in os.listdir(ce_path)]
        self.cases = np.random.permutation(np.array(self.cases))

        assert (len(self.cases) > 0)
        self.patch_size = patch_size
        self.is_foldsplit = False
        self.is_train = is_train
        self.blur_kernel = blur_kernel
        self.patches_per_case = patches_per_case
        self.case_patch_count = {case: 0 for case in self.cases}
        self.patch_per_val_case=10


    def apply_foldsplit(self, split_ratio=0.8, train_cases=None):
        if type(train_cases) == type(None):
            self.train_cases = np.random.choice(self.cases, int(split_ratio * len(self.cases)), replace=False)
        else:
            self.train_cases = train_cases

        self.test_cases = self.cases[~np.isin(self.cases, self.train_cases)]
        self.is_foldsplit = True

    def __len__(self):
        assert (self.is_foldsplit)

        if self.is_train:
            return len(self.train_cases)*self.patches_per_case
        else:
            return len(self.test_cases) * self.patch_per_val_case


    def _add_noise(self, tensor1, tensor2, p=0.3, noise_strength=0.3):
        if random() > p: return tensor1, tensor2
        random_noise_stdev = random() * noise_strength
        noise = torch.randn(tensor1.shape) * random_noise_stdev
        return tensor1 + noise, tensor2

    def _rotate(self, tensor1, tensor2, p=0.3):
        if random() > p: return tensor1, tensor2
        rot_extent = np.random.randint(1, 4)
        return torch.rot90(tensor1, rot_extent, dims=[-2, -1]), torch.rot90(tensor2, rot_extent, dims=[-2, -1])

    def _flip(self, tensor1, tensor2, p=0.3):
        if random() > p: return tensor1, tensor2
        flip = int(np.random.choice([-3, -2, -1], 1, replace=False))
        return torch.flip(tensor1, dims=[flip]),torch.flip(tensor2, dims=[flip])

    def _blur(self, tensor1, tensor2, p=0.3):
        if random() > p: return tensor1, tensor2
        return self.blur_kernel(tensor1), tensor2

    def _contrast(self, tensor1, tensor2, p = 0.2):
        if random() > p: return tensor1, tensor2

        fac = np.random.uniform(*[0.9, 1.1])
        mean = tensor1.mean()
        mn = tensor1.min().item()
        mx = tensor1.max().item()
        tensor1 = (tensor1 - mean) * fac + mean

        return tensor1.clip(mn, mx), tensor2

    def reset_patch_count(self):
        self.case_patch_count = {case: 0 for case in self.cases}

    def _get_random_patch_indices(self, image1,image2, patch_size):
        '''

        Parameters
        ----------
        image1, image2 : torch.Tensor
            3D image to take a random patch from
        patch_size
            size of the patch to take from the image
        Returns
        -------
        indices : tuple
            indices to take the patch from the image
        '''
        # delete all training data where image1 and image2 are not the same size within 3 pixels,
        # or there is inconsistent slice interval (n=10)
        for dim in range(1,4):
            assert (image1.shape[dim] - image2.shape[dim])<4
        # get the shape of the image
        shape = [min(image1.shape[dim], image2.shape[dim]) for dim in range(4)]
        # pad the image to the size of the patch if the image is smaller than the patch - use torch padding
        padding = []
        for dim in range(1,4):
            if shape[dim] < patch_size:
                pad = patch_size - shape[dim]
                padding = padding +[0, pad]
            else:
                padding = padding +[0, 0]

        image1 = torch.nn.functional.pad(image1, tuple(padding[::-1]),'constant',-3.15)
        image2 = torch.nn.functional.pad(image2, tuple(padding[::-1]),'constant',-3.15)
        shape = [min(image1.shape[dim], image2.shape[dim]) for dim in range(4)]


        # get the random indices to take the patch from
        xyz = []
        for dim in range(1,4):
            if shape[dim] == patch_size:
                xyz.append(0)
            else:
                xyz.append(np.random.randint(0, shape[dim] - patch_size))

        return image1,image2,tuple(xyz)

    def __getitem__(self, idx: int):
        assert (self.is_foldsplit)
        if self.is_train:
            transforms = [self._blur, self._add_noise, self._rotate, self._rotate, self._rotate,
                          self._flip, self._flip,self._flip, self._contrast]
            # possible cases are those in the training cases that have not yet had 10 patches taken
            possible_cases = [case for case in self.train_cases if self.case_patch_count[case] <= self.patches_per_case]
            case = possible_cases[idx % len(possible_cases)]
            self.case_patch_count[case] += 1
        else:
            transforms = None
            case = self.test_cases[idx % (len(self.test_cases)-1)]


        ne = torch.Tensor(np.load(os.path.join(self.ne_path, case)))
        ce = torch.Tensor(np.load(os.path.join(self.ce_path, case)))
        # get random patch indices
        ne,ce,indices = self._get_random_patch_indices(ne,ce, self.patch_size)
        ne_patch = ne[0, indices[0]:indices[0]+self.patch_size, indices[1]:indices[1]+self.patch_size, indices[2]:indices[2]+self.patch_size]
        ce_patch = ce[0, indices[0]:indices[0]+self.patch_size, indices[1]:indices[1]+self.patch_size, indices[2]:indices[2]+self.patch_size]

        if self.is_train:
            if transforms:
                # shuffles order of transforms every time
                np.random.shuffle(transforms)
                for transform in transforms:
                    ne_patch,ce_patch = transform(ne_patch,ce_patch)

        return ne_patch.unsqueeze(0), ce_patch.unsqueeze(0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
    dataset_ne_path = os.path.join(
        '/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/small_coreg_ncct/small_coreg_ncct_2/images')
    dataset_ce_path = os.path.join('/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/add_cect/add_cect_2/images')
    # test the dataset
    # ce_path = '/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/coltea_art/coltea_art_2/images/'
    # ne_path = '/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/coltea_nat/coltea_nat_2/images/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    dataset = CrossPhaseDataset(dataset_ne_path, dataset_ce_path, device=device, is_train=True)

    dataset.apply_foldsplit()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (ne, ce) in enumerate(dataloader):
        fig, ax = plt.subplots(2, 2)
        ax[0,0].imshow(ne[0, 0, 63, ].cpu().detach().numpy())
        ax[0,1].imshow(ce[0, 0, 63, ].cpu().detach().numpy())
        ax[1,0].imshow(ne[0, 0, 0,].cpu().detach().numpy())
        ax[1,1].imshow(ce[0, 0, 0,].cpu().detach().numpy())
        plt.show(block=True)