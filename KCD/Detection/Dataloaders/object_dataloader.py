#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:59:32 2023

@author: mcgoug01
"""

import os
import torch
from torch.utils.data import Dataset
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import dgl

def assign_rowwise_max(df, columns,new_column_name):
    df[new_column_name] = df[columns].max(axis=1)
    return df

def convert_to_binary(df, column_name, substring):
    df[column_name] = df[column_name].str.contains(substring).astype(int)
    return df

class ObjectData_labelled(Dataset):
    def __init__(self, home,data_name='coreg_ncct',
                 graph_thresh=1000, mlp_thresh=20000,
                 voxel_spacing_mm=1, thresh_r_mm=10,dev='cpu'):
        
        self.graph_thresh = graph_thresh
        self.mlp_thresh = mlp_thresh
        
        self.homepath = os.path.join(home,'objects')
        print(self.homepath)
        assert(os.path.exists(self.homepath))
        assert(data_name!=None)

        ########## OBJECT PROPERTIES ###########
        # load filepaths
        self.edges = os.path.join(self.homepath,data_name,'edges')
        self.curvatures = os.path.join(self.homepath,data_name,'curvatures')
        self.vertices = os.path.join(self.homepath,data_name,'vertices')
        
        ########## DATA PREPROCESSING ###########
        self.case_data = pd.read_csv(os.path.join(self.homepath,data_name,'features_labelled.csv'),index_col=False)
        # filter out central kidneys
        central_kids = (self.case_data.position=='centre') | (self.case_data.position=='central')
        self.case_data = self.case_data[~central_kids]
        
        self.params = pd.read_csv(os.path.join(self.homepath,data_name,'normalisation_params.csv'),index_col=False)
        self.case_data = assign_rowwise_max(self.case_data,columns = [col for col in self.case_data.columns if col.startswith('cancer_')],
                                                              new_column_name='largest_cancer')
        self.case_data = assign_rowwise_max(self.case_data,columns = [col for col in self.case_data.columns if col.startswith('cyst_')],
                                                              new_column_name='largest_cyst')
        
        for col in [column for column in self.case_data.columns if (is_numeric_dtype(self.case_data[column]))]:
            if (col.endswith('_vol')) or ('Unnamed' in col): self.case_data= self.case_data.drop(col,axis=1)
            elif not (col in self.params.col.values):continue
            else:
                mean,std = self.params[self.params['col'] == col]['mean'].values[0],self.params[self.params['col'] == col]['std'].values[0]
                self.case_data[col] = (self.case_data[col]-mean)/std 
        
        # load case-specific filepaths
        self.case_data['filename'] = (self.case_data['case'].str.split('.').str[0] + '_'+self.case_data['position']).values
        self.case_data['case'] = self.case_data['case'].str.split('.').str[0].replace('-','_')
        self.case_data['obj_fp'] = os.path.join(self.homepath,data_name,'cleaned_objs')+'/'+self.case_data['case']+ '_'+self.case_data['position']+'.obj'
        self.cases = self.case_data.case
        
        # convert position string to binary feature
        self.case_data = convert_to_binary(self.case_data,'position','right')
        self.case_data['label'] = (self.case_data['largest_cyst']>self.graph_thresh) | (self.case_data['largest_cancer']>self.graph_thresh)

        ########## DATASET PROPERTIES ###########

        benign,malign = len(self.case_data.label) - sum(self.case_data.label),sum(self.case_data.label)
        print("Graph training data contains {} normal cases and {} misshapen cases.".format(benign,malign))

        self.device=dev
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
    
    def _get_graph(self,case_df):
        
        fp = case_df.filename +'.npy'
        
        vertices = np.load(os.path.join(self.vertices,fp))
        vertices = (vertices - vertices.mean(axis=0))/5
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
    
    
    
class ObjectData_unlabelled(Dataset):
    def __init__(self, home,data_name='coreg_ncct',norm_param_dataname='coreg_ncct',
                 graph_thresh=1000, mlp_thresh=20000,
                 voxel_spacing_mm=1, thresh_r_mm=10,dev='cpu'):
        
        self.graph_thresh = graph_thresh
        self.mlp_thresh = mlp_thresh
        
        self.homepath = os.path.join(home,'objects')
        print(self.homepath)
        assert(os.path.exists(self.homepath))
        assert(data_name!=None)

        ########## OBJECT PROPERTIES ###########
        # load filepaths
        self.edges = os.path.join(self.homepath,data_name,'edges')
        self.curvatures = os.path.join(self.homepath,data_name,'curvatures')
        self.vertices = os.path.join(self.homepath,data_name,'vertices')
        
        ########## DATA PREPROCESSING ###########
        self.case_data = pd.read_csv(os.path.join(self.homepath,data_name,'features_unlabelled.csv'))
        # filter out central kidneys
        central_kids = (self.case_data.position=='centre') | (self.case_data.position=='central')
        self.case_data = self.case_data[~central_kids]
        
        self.params = pd.read_csv(os.path.join(self.homepath,norm_param_dataname,'normalisation_params.csv'))
        
        for col in [column for column in self.case_data.columns if (is_numeric_dtype(self.case_data[column]))]:
            if (col.endswith('_vol')): self.case_data= self.case_data.drop(col,axis=1)
            elif not (col in self.params.col.values):continue
            else:
                mean,std = self.params[self.params['col'] == col]['mean'].values[0],self.params[self.params['col'] == col]['std'].values[0]
                self.case_data[col] = (self.case_data[col]-mean)/std 
        
        # load case-specific filepaths
        self.case_data['filename'] = (self.case_data['case'].str.split('.').str[0] + '_'+self.case_data['position']).values
        self.case_data['case'] = self.case_data['case'].str.split('.').str[0].replace('-','_')
        self.case_data['obj_fp'] = os.path.join(self.homepath,data_name,'cleaned_objs')+'/'+self.case_data['case']+ '_'+self.case_data['position']+'.obj'
        self.cases = self.case_data.case
        
        # convert position string to binary feature
        self.case_data = convert_to_binary(self.case_data,'position','right')

        ########## DATASET PROPERTIES ###########

        self.device=dev
        self.test_case = None
        self.is_casespecific = False
        
    def __len__(self):
        assert(self.is_foldsplit)
        if self.is_casespecific:
            return len(self.case_specific_data)
        else: 
            return len(self.case_data)
        
    def set_val_kidney(self,case:str,side='left'):
        self.is_casespecific = True
        self.test_case = case
        self.case_specific_data = self.case_data[(self.case_data['case'] == self.test_case) & (self.case_data['position']==side)]
        if len(self.case_specific_data)==0:
            print("You tried to set the validation kidney to a kidney that does not exist within the validation data.")
            assert(len(self.case_specific_data)>0)
        assert(len(self.case_specific_data)==1)
    

    def _get_graph(self,case_df):
        fp = case_df.filename +'.npy'
        vertices = np.load(os.path.join(self.vertices,fp))
        vertices = (vertices - vertices.mean(axis=0))/5
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
        features = case_df.drop(['obj_fp','filename','case']).values
        
        return torch.Tensor(features.astype(float))
    
    
    def __getitem__(self,idx:int):
        if not self.is_casespecific: case_df = self.case_data.iloc[idx]
        else: case_df = self.case_specific_data.iloc[0]
            
        features = self._get_features(case_df.copy())
        graph = self._get_graph(case_df.copy())
        
        return features.to(self.device),graph.to(self.device)
    
    

if __name__ == "__main__":
    dataset='merged_training_set'
    name = 'add_ncct_unseen'
    ds1 = ObjectData_unlabelled('/Users/mcgoug01/Downloads/Data',data_name='add_ncct_unseen',norm_param_dataname=name)
    a = ds1[0]
    
    # ds2 = ObjectData_labelled('/Users/mcgoug01/Downloads/Data',data_name=dataset)
    # ds2.apply_foldsplit()
    # b = ds2[0]