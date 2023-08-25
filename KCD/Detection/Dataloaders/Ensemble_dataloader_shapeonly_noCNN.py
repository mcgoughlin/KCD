#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:59:32 2023

@author: mcgoug01
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from random import random
import dgl
import torchvision

class EnsembleDataset(Dataset):
    def __init__(self, obj_path,data_name='kits23sncct',
                 graph_thresh=1000, mlp_thresh=20000,
                 voxel_spacing_mm=1, thresh_r_mm=10, is_masked=True,
                 spacing='4',dev='cpu',transform= True):
        
        self.spacing = int(spacing)
        self.graph_thresh = graph_thresh
        self.mlp_thresh = mlp_thresh
        
        self.homepath = obj_path
        print(self.homepath)
        assert(os.path.exists(self.homepath))
        assert(data_name!=None)

        ########## OBJECT PROPERTIES ###########
        # load filepaths
        self.edges = os.path.join(self.homepath,data_name,'edges_{}mm'.format(spacing))
        self.curvatures = os.path.join(self.homepath,data_name,'curvatures_{}mm'.format(spacing))
        self.vertices = os.path.join(self.homepath,data_name,'vertices_{}mm'.format(spacing))
        
        # load case data, clean any erroneous 'unnamed' columns caused as artefact of csv loading/saving
        self.case_data = pd.read_csv(os.path.join(self.homepath,data_name,'preprocessed_features_{}mm.csv'.format(spacing)),index_col=False)
        for col in self.case_data.columns:
            if 'Unnamed' in col: self.case_data= self.case_data.drop(col,axis=1)
                
        # load case-specific filepaths
        self.case_data['filename'] = (self.case_data['case'].str.split('.').str[0] + '_'+self.case_data['position']).values
        self.case_data['case'] = self.case_data['case'].str.split('.').str[0].replace('-','_')
        self.case_data['obj_fp'] = os.path.join(self.homepath,data_name,'cleaned_objs_{}mm'.format(spacing))+'/'+self.case_data['case']+ '_'+self.case_data['position']+'.obj'

        # filter out central kidneys
        central_kids = (self.case_data.position=='centre') | (self.case_data.position=='central')
        self.case_data = self.case_data[~central_kids]

        # ensure each case is represented by both the tile-wise data and object-wise data
        self.obj_cases = self.case_data['case'].unique()
        self.cases = self.case_data.case
        
        self.case_data['label'] = (self.case_data['largest_cyst']>self.graph_thresh) | (self.case_data['largest_cancer']>self.graph_thresh)

        ########## DATASET PROPERTIES ###########

        benign,malign = len(self.case_data.label) - sum(self.case_data.label),sum(self.case_data.label)
        print("Graph training data contains {} normal cases and {} misshapen cases.".format(benign,malign))

        self.device=dev
        self.transform = transform
        self.blur_kernel = torchvision.transforms.GaussianBlur(3)
        self.is_foldsplit = False
        self.test_case = None
        self.is_train = True
        self.is_casespecific = False
        
    def __len__(self):
        assert(self.is_foldsplit)
        if self.is_casespecific:
            return len(self.case_specific_data)
        else: 
            if self.is_train:
                return len(self.train_data)
            else:
                return len(self.test_data)
        
    def set_val_kidney(self,case:str,side='left'):
        self.is_casespecific = True
        self.test_case = case
        self.case_specific_data = self.case_data[(self.case_data['case'] == self.test_case) & (self.case_data['position']==side)]
        if len(self.case_specific_data)==0:
            print("You tried to set the validation kidney to a kidney that does not exist within the validation data.")
            assert(len(self.case_specific_data)>0)
        assert(len(self.case_specific_data)==1)
        
    
    
    def apply_foldsplit(self,split_ratio=0.8,train_cases=None):
        if type(train_cases)==type(None):
            self.train_cases = np.random.choice(np.unique(self.cases),int(split_ratio*len(np.unique(self.cases))),replace=False)
        else:
            self.train_cases = train_cases
            
        self.test_cases = np.unique(self.cases[~np.isin(self.cases, self.train_cases)])
        self.train_data = self.case_data[self.case_data['case'].isin(self.train_cases)]
        self.test_data = self.case_data[~np.isin(self.case_data['case'],self.train_cases)]
        
        self.is_foldsplit=True
    
    def _add_noise(self,tensor,p=0.3):
        if random()>p: return tensor
        random_noise_stdev = random()*0.1
        noise = torch.randn(tensor.shape,device=self.device)*random_noise_stdev
        return tensor+noise
    
    def _rotate(self,tensor,p=0.5):
        if random()>p: return tensor
        rot_extent = int(np.random.choice([1,2,3]))
        rot =torch.rot90(tensor,rot_extent,dims=(1,2))
        return rot
    
    
    def _flip(self,tensor,p=0.5):
        if random()>p: return tensor
        return torch.flip(tensor,dims= [int(np.random.choice([-2,-1],1,replace=False))])
        
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
    
    def _get_graph(self,case_df):
        
        fp = case_df.filename +'.npy'
        
        vertices = np.load(os.path.join(self.vertices,fp))
        vertices = (vertices - vertices.mean(axis=0))/(5*4/self.spacing)
        # vertices = (vertices)/50

        edges = np.load(os.path.join(self.edges,fp))
        curvatures = np.load(os.path.join(self.curvatures,fp))

        # source nodes
        u = edges[:,0]
        # destination nodes
        v = edges[:,1]
        # Tuple of tensors is passed as argument, and edges are made bidrectional
        mygraph = dgl.to_bidirected(dgl.graph((u, v)))
         
        # followingthis tutorial - https://docs.dgl.ai/en/0.2.x/tutorials/basics/1_first.html

        input_x = torch.Tensor(np.array([*vertices.T,curvatures]).T)
        mygraph.ndata['feat'] = input_x

        return mygraph
    
    def _get_features(self,case_df):
        
        
        case_df['pos_binary'] = int(case_df['position'] =='right')
        features = case_df.drop(['label','case','position','largest_cyst',
                                 'obj_fp','filename','largest_cancer',
                                 'case']).values
        
        return torch.Tensor(features.astype(float))
    
    
    def __getitem__(self,idx:int):
        assert(self.is_foldsplit)
        
        if not self.is_casespecific:
            if self.is_train:
                
                idx = idx%len(self.train_data)
                case_df = self.train_data.iloc[idx]
            else:
                idx = idx%len(self.test_data)
                case_df = self.test_data.iloc[idx]
        else:
            case_df = self.case_specific_data.iloc[0]
            
            
        label_list = []
        for thresh in [self.mlp_thresh,self.graph_thresh]:
            obj_is_cancerous = int(case_df.largest_cancer>thresh)
            obj_is_cystic = int(case_df.largest_cyst>thresh) 
            obj_label = max(obj_is_cystic,obj_is_cancerous,0)
            label_list.append(obj_label)
            
        features = self._get_features(case_df.copy())
        graph = self._get_graph(case_df.copy())
        
        return features.to(self.device),graph.to(self.device), torch.Tensor(label_list).long().to(self.device)

if __name__ == "__main__":
    dataset = EnsembleDataset('/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/',
                              '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/CNN_dataset/SimpleView')
    dataset.apply_foldsplit()
    sv_im,features,graph, label = dataset.__getitem__(3)
    pass