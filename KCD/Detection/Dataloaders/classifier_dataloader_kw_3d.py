# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:16:53 2022

@author: mcgoug01
"""
from os import *
from os.path import *
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from random import random
from scipy.ndimage import gaussian_filter


#create and instantiate data-loading class

##datapath goes up to the mm folder, under which should be "kidneys","tumour","none"
class SW_Data3d(Dataset):
    def __init__(self, path, voxel_spacing_mm, thresh_r_mm,depth=20,data_name=None, 
                 transform= True,device=None,
                 epoch_length = 1000,batch_size = 12,is_masked=False):
        print(thresh_r_mm)
        assert(exists(path))
        assert(data_name!=None)
        
        if is_masked: raw_data_path = join(path,"sliding_window_kidneywisemasked",data_name)
        else: raw_data_path = join(path,"sliding_window_kidneywise",data_name)
        assert(exists(raw_data_path))
        
        voxel_path = join(raw_data_path,"Voxel-"+str(voxel_spacing_mm)+"mm")
        assert(exists(voxel_path))
        
        home = join(voxel_path,"Threshold-"+str(thresh_r_mm)+"mm",'3D-Depth{}mm'.format(depth))
        print(home)
        assert(exists(home))
        assert([folder in listdir(home)for folder in ["tumour","kidney"]])
        self.home_path = home
        self.device = device

        self.epoch_length = epoch_length
        self.epoch = 0
        
        self.malign = listdir(join(home,"tumour"))
        self.benign = listdir(join(home,"kidney"))
        self.none = listdir(join(home,"none"))

        self.data_df = pd.DataFrame(data=np.array([[*self.none,*self.malign,*self.benign],
                                          [*[0]*len(self.none), *[2]*len(self.malign), *[1]*len(self.benign)]]).T,
                                    columns = ['filepath','class'])
        self.data_df['class'] = self.data_df['class'].astype(int)
        self.data_df['case'] = self.data_df.filepath.str.replace('RCC_','RCC-').str.split('_').apply(lambda x:x[0])
        self.data_df['side'] = self.data_df.filepath.str.replace('RCC_','RCC-').str.split('_').apply(lambda x:x[1])
        self.data_df['window'] = self.data_df.filepath.str.replace('RCC_','RCC-').str.split('_').apply(lambda x:x[2])
        self.data_df['slice'] = self.data_df.filepath.str.replace('RCC_','RCC-').str.split('_').apply(lambda x:int(x[-1].split('index')[1].split('.')[0]))
        self.cases = self.data_df.case.unique()
            
        self.data_dict = {0:self.none,1:self.benign,2:self.malign}
        self.dir_dict = {0:join(self.home_path,"none"),
                         1:join(self.home_path,"kidney"),
                         2:join(self.home_path,"tumour")}        
        benign,malign = len(self.benign),len(self.malign)
        print("Training data contains {} benign cases and {} malign cases.".format(benign,malign))
                        
        self.batch_size=batch_size
        self.transform = transform
        self.blur_kernel = torchvision.transforms.GaussianBlur(3)
        self.is_foldsplit = False
        self.is_train = True
        self.test_case = None

    def __len__(self):
        assert(self.is_foldsplit)

        if type(self.test_case) == type(None):
            if self.is_train: return 3*max(self.train_data['class'].value_counts())
            else: return len(self.test_data)
        else:
            return len(self.case_specific_data)
        
    def set_val_kidney(self,case:str,side='left'):
        self.test_case = case
        self.case_specific_data = self.data_df[(self.data_df['case'] == self.test_case) & (self.data_df['side']==side) & (self.data_df['window']=='centralised')]
        self.case_specific_data = self.case_specific_data.sort_values('slice')
        if len(self.case_specific_data)==0:
            print("You tried to set the validation kidney to a kidney that does not exist within the validation data.")
            assert(len(self.case_specific_data)>0)
    
    def _add_noise(self,tensor,p=0.3,noise_strength=0.3):
        if random()>p: return tensor
        random_noise_stdev = random()*noise_strength
        noise = torch.randn(tensor.shape,device=self.device)*random_noise_stdev
        return tensor+noise
    
    def _rotate(self,tensor,p=1):
        if random()>p: return tensor
        rot_extent = np.random.randint(1,4)
        return torch.rot90(tensor,rot_extent,dims=[-2,-1])
    
    def _flip(self,tensor,p=0.5):
        if random()>p: return tensor
        flip = int(np.random.choice([-3,-1],1,replace=False))
        return torch.flip(tensor,dims= [flip])
        
    def _blur(self,tensor,p=0.3):
        if random()>p: return tensor
        return self.blur_kernel(tensor)
        
    def _contrast(self, img):
        fac = np.random.uniform(*[0.9,1.1])
        mean = img.mean()
        mn = img.min().item()
        mx = img.max().item()
        img = (img - mean) * fac + mean
        
        return img.clip(mn, mx)
    
    def apply_foldsplit(self,split_ratio=0.8,train_cases=None):
        if type(train_cases)==type(None):
            self.train_cases = np.random.choice(self.cases,int(split_ratio*len(self.cases)),replace=False)
            
        else:
            self.train_cases = train_cases
            
        self.test_cases = self.cases[~np.isin(self.cases, self.train_cases)]
        self.train_data = self.data_df[self.data_df['case'].isin(self.train_cases)]
        self.test_data = self.data_df[~np.isin(self.data_df['case'],self.train_cases)]
        
        self.is_foldsplit=True
    
    def __getitem__(self,idx:int):
        assert(self.is_foldsplit)
        transforms = []
        if type(self.test_case) == type(None):
            if self.is_train:
                transforms = [self._blur,self._add_noise,self._rotate,self._flip,self._contrast,self._flip,self._flip]
                label = int(np.random.choice([0,1,2],size=1))
                class_df = self.train_data[self.train_data['class']==label]
                idx = idx % len(class_df)
            else:
                class_df = self.test_data
                idx = idx % len(class_df)
                label = class_df.iloc[idx]['class']
        else:
            class_df = self.case_specific_data
            label = class_df.iloc[idx]['class']
            
        directory = self.dir_dict[label]
        image = np.load(join(directory,class_df.iloc[idx]['filepath']))

        image = torch.clip(torch.Tensor(image).to(self.device),-200,200)/100
        print(image.shape)
        image=image[0]
        
        #shuffles order of transforms every time
        np.random.shuffle(transforms)
        for transform in transforms:
            image = transform(image)
            
        print(image.shape)
            
        return image,label
    
def get_dataloader(home,voxel_spacing=2,data_name='test',
                   is_3D=False, batch_size = 16):
    dataset = SW_Data3d(home,voxel_spacing,data_name=data_name,is_3D=is_3D)        
    return DataLoader(dataset,batch_size=batch_size,shuffle=True)

def get_dataset(home,voxel_spacing=2,thresh_r_mm = 5,data_name='test',
                batch_size = 16,depth=20,is_masked=False):
    return SW_Data3d(home,voxel_spacing,thresh_r_mm,data_name=data_name,depth=depth,is_masked=is_masked)  

if __name__ == '__main__':
    home = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/CNN_dataset"
    dataset = get_dataset(home,voxel_spacing=1,thresh_r_mm=10,depth=20,data_name='coreg_ncct',is_masked=True)
    dataset.apply_foldsplit()
    dataset.is_train=True
    import matplotlib.pyplot as plt
    a,b = dataset[np.random.randint(0,3000)]
    print(b)

    for slice in a[0]:
        fig = plt.figure(figsize=(6,6))
        plt.imshow(slice,vmax=2.5,vmin=-2.5)
        plt.show(block=True)
    # plt.subplot(122)
    # plt.imshow(a[0,-1],vmax=2.5,vmin=-2.5)
    # plt.show()
    # dataset.is_train=False
    # val_kid = 'KiTS-00114'
    # print("Val Kidney is {}".format(val_kid))
    # dataset.set_val_kidney(val_kid,'right')
    import matplotlib.pyplot as plt
    # for i in range(len(dataset)):
    #     a,b = dataset[i]
    #     fig = plt.figure(figsize=(8,8))
    #     plt.imshow(a[0])
    #     plt.show(block=True)