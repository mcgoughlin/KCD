import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
from random import random

#make two dataloaders - one for labelled, one for unlabelled 
##datapath goes up to the mm folder, under which should be "kidneys","tumour","none"
class SW_Data_seglabelled(Dataset):
    def __init__(self, path, name,voxel_size_mm=1,cancthresh=10,kidthresh=20,
                 depth_z=20,boundary_z=5,dilated=40,device=None):
        assert(os.path.exists(path))

        raw_data_path = os.path.join(path,"slices",name)
        seg_data_path = os.path.join(path, "segs", name)

        assert(os.path.exists(raw_data_path))
        assert(os.path.exists(seg_data_path))
        home = []
        for path in [raw_data_path,seg_data_path]:
            folder = os.path.join(path,"Voxel-"+str(voxel_size_mm)+"mm-Dilation"+str(dilated)+"mm")
            folder = os.path.join(folder,"CancerThreshold-"+str(cancthresh)+'mm-KidneyThreshold'+str(kidthresh)+'mm')
            if depth_z>1:
                home.append(os.path.join(folder,"3D-Depth"+str(depth_z)+'mm-Boundary'+str(boundary_z)+'mm','labelled_data'))
                self.is_3d = True
            else:
                home.append(os.path.join(folder,"2D-Boundary"+str(boundary_z)+'mm','labelled_data'))
                self.is_3d = False
        
        self.image_home = home[0]
        self.seg_home = home[1]
        print(self.image_home)
        assert(os.path.exists(self.image_home))
        assert([folder in os.listdir(self.image_home)for folder in ["tumour","kidney","none"]])

        self.device = device
        
        self.malign = os.listdir(os.path.join(self.seg_home,"tumour"))
        self.benign = os.listdir(os.path.join(self.seg_home,"kidney"))
        self.none = os.listdir(os.path.join(self.seg_home,"none"))

        self.data_df = pd.DataFrame(data=np.array([[*self.none,*self.malign,*self.benign],
                                          [*[0]*len(self.none), *[2]*len(self.malign), *[1]*len(self.benign)]]).T,
                                    columns = ['filepath','class'])
        
        self.data_df['class'] = self.data_df['class'].astype(int)
        self.data_df['case'] = self.data_df.filepath.str.replace('-','_').str.split('_').apply(lambda x:x[0:2]).str.join("_")
        self.data_df['side'] = self.data_df.filepath.str.replace('-','_').str.split('_').apply(lambda x:x[2])
        self.data_df['window'] = self.data_df.filepath.str.replace('-','_').str.split('_').apply(lambda x:x[3])
        self.data_df['slice'] = self.data_df.filepath.str.replace('-','_').str.split('_').apply(lambda x:int(x[-1].split('index')[1].split('.')[0]))
        self.cases = np.unique(self.data_df.case.values)
                    
        self.data_dict = {0:self.none,1:self.benign,2:self.malign}
        self.dir_dict = {0:(os.path.join(self.image_home,"none"),os.path.join(self.seg_home,"none")),
                         1:(os.path.join(self.image_home,"kidney"),os.path.join(self.seg_home,"kidney")),
                         2:(os.path.join(self.image_home,"tumour"),os.path.join(self.seg_home,"tumour"))}
        benign,malign,none = len(self.benign),len(self.malign),len(self.none)
        print("Training data contains {} benign slices, {} none slices, and {} malign slices.".format(benign,none,malign))
                        
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
    
    def _add_noise(self,im,seg,p=0.3,noise_strength=0.3):
        if random()>p: return im,seg
        random_noise_stdev = random()*noise_strength
        noise = torch.randn(im.shape,device=self.device)*random_noise_stdev
        return im+noise, seg
    
    def _rotate(self,tensor,seg,p=0.5):
        if random()>p: return tensor,seg
        rot_extent = np.random.randint(1,4)
        return torch.rot90(tensor,rot_extent,dims=[-2,-1]), torch.rot90(seg,rot_extent,dims=[-2,-1])
    
    def _flip(self,im,seg,p=0.5):
        if random()>p: return im,seg
        if self.is_3d:
            flip = int(np.random.choice([-3,-2,-1],1,replace=False)[0])
        else:
            flip = int(np.random.choice([-2,-1],1,replace=False)[0])
        return torch.flip(im,dims= [flip]), torch.flip(seg,dims= [flip])
        
    def _blur(self,im,seg,p=0.3):
        if random()>p: return im,seg
        return self.blur_kernel(im),seg
        
    def _contrast(self, img,seg):
        fac = np.random.uniform(*[0.9,1.1])
        mean = img.mean()
        mn = img.min().item()
        mx = img.max().item()
        img = (img - mean) * fac + mean
        
        return img.clip(mn, mx),seg
    
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
                transforms = [self._blur,self._add_noise,self._rotate,self._flip,self._contrast,self._flip,self._rotate]
                label = int(np.random.choice([0,1,2],size=1)[0])
                class_df = self.train_data[self.train_data['class']==label]
                idx = idx % len(class_df)
            else:
                class_df = self.test_data
                idx = idx % len(class_df)
                label = class_df.iloc[idx]['class']
        else:
            class_df = self.case_specific_data
            label = class_df.iloc[idx]['class']
            
        directory,seg_dir = self.dir_dict[label]
        image = np.load(os.path.join(directory,class_df.iloc[idx]['filepath']),allow_pickle=True)
        image = torch.clip(torch.Tensor(image).to(self.device),-200,200)/100
        image = torch.unsqueeze(image,0)
        seg = np.load(os.path.join(seg_dir,class_df.iloc[idx]['filepath']),allow_pickle=True)
        seg = torch.Tensor(seg).to(self.device)

        #shuffles order of transforms every time
        np.random.shuffle(transforms)
        for transform in transforms:
            image,seg = transform(image,seg)

        return image,(torch.Tensor(seg).long(),torch.Tensor([label]).long())

if __name__ == '__main__':
    dataset = SW_Data_seglabelled(path="/Users/mcgoug01/Downloads/Data",name="coreg_ncct",device='cpu')
    dataset.apply_foldsplit()
    im,(seg,label) = dataset[0]

    # plot im and seg with reduce alpha
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(im[0,0,:,:])
    plt.imshow(seg[0,0,:,:],alpha=0.5)
    plt.show(block=True)