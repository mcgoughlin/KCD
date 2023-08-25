# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:12:22 2022

@author: mcgoug01
"""
import Ensemble_dataloader_shapeonly_infer as dl_shape
import tilewise_dataloader as tw_dl
import infer_methods as infer

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import dgl
import warnings
import pandas as pd

if torch.cuda.is_available():
    dev = 'cuda'
else:
    dev = 'cpu'

def shape_collate(samples,dev=dev):
    # The input `samples` is a list of pairs
    #  (sv_im,features,graph, obj_label).
    svim,features,graphs, labels = map(list, zip(*samples))

    batched_graph = dgl.batch(graphs)
    return torch.stack(svim).to(dev),torch.stack(features).to(dev),batched_graph.to(dev), torch.stack(labels).squeeze(dim=0).to(dev)

def compute_loss_weighting(spec_multiplier,num_classes=2):
    sens = num_classes/(spec_multiplier+1)
    spec = spec_multiplier*sens
    
    return sens,spec

# tilewise CNN
vox_spacing = 1
thresh_r =10
tw_batch_size= 32
tw_lr =  0.0005
tw_size='large'
voting_size=10

#### ensemble model opt. params
s1_CNNepochs = 5

#### data params
tile_dataname = 'add_ncct_unseen'
shape_ensemble_dataname ='add_ncct_unseen'

#### ignore all warnings - due to dgl being very annoying

ignore = True
if ignore: warnings.filterwarnings("ignore")

#### path init 
 
save_dir = '/home/wcm23/rds/hpc-work/EnsembleResults/ensemble_models'
load_dir = '/home/wcm23/rds/hpc-work/EnsembleResults/saved_info'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir,'csv'))
    os.mkdir(os.path.join(save_dir,'png'))


obj_path = '/home/wcm23/rds/hpc-work/GraphData/'
tile_path = '/home/wcm23/rds/hpc-work/'
simpleview_path = '/home/wcm23/rds/hpc-work/SimpleView'

#### training housekeeping
results=[] 
softmax = nn.Softmax(dim=-1)

#### init all datasets needed 

test_shapedataset = dl_shape.EnsembleDataset(obj_path,
                         simpleview_path,
                         data_name=shape_ensemble_dataname,
                         graph_thresh=0,mlp_thresh=0,sv_thresh=0)

test_tilewise_dataset = tw_dl.get_dataset(tile_path,voxel_spacing=vox_spacing,thresh_r_mm=0,data_name=tile_dataname,is_masked=True)


# highlight cases that are authentic ncct as 1, others as 0
# this allows the k-fold case allocation to evenly split along the authentic ncct class
cases = np.unique(test_shapedataset.cases.values)
twname = 'resnext_{}_{}_{}_{}'.format(tw_size,s1_CNNepochs,tw_batch_size,tw_lr)
tw_boundary = []

for reading in [0]:
    for split in [0]:
        split_path = os.path.join(load_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)
            
        split_fp = os.path.join(split_path,'split.npy')
        if os.path.exists(split_fp):
            fold_split = np.load(split_fp,allow_pickle=True)
        else:  
            assert(1==2)
        
        for fold in range(5):
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            twCNN_path = os.path.join(fold_path,'twCNN')
            twCNN_results = pd.read_csv(os.path.join(twCNN_path,'csv',twname+'_{}.csv'.format(reading)))
            twCNN_results = twCNN_results[(twCNN_results['dataset_loc']=='test')&(twCNN_results['Voting Size']==voting_size)]
            tw_boundary.append(twCNN_results['Boundary 98'].values[0])
            print(tw_boundary)
            
tw_boundary =np.median(tw_boundary)
print('boundary is {}.'.format(tw_boundary))

# begin training!
for reading in [0]:
    for split in [0]:
        split_path = os.path.join(load_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)
            
        split_fp = os.path.join(split_path,'split.npy')
        if os.path.exists(split_fp):
            fold_split = np.load(split_fp,allow_pickle=True)
        else:  
            assert(1==2)
        
        for fold in range(5):
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            twCNN_path = os.path.join(fold_path,'twCNN')
            
            twCNN_fp = os.path.join(twCNN_path,'model',twname)
            twCNN = torch.load(twCNN_fp,map_location=dev)
        
            print("\nFold {} training.".format(fold))
            
            ######## Individual Object Model Eval ########
        
            test_shapedataset.apply_foldsplit(train_cases = cases)
            test_dl = DataLoader(test_shapedataset,batch_size=8,shuffle=True,collate_fn=shape_collate)
            print('cases',test_dl.dataset.train_cases)
                        
            ######## Individual CNN Model Eval ########
            
            print("Evaluating tile classifier")
            
            test_tilewise_dataset.apply_foldsplit(train_cases = cases)
            test_tilewise_dataset.is_train=False            
            test_tw_dl = DataLoader(test_tilewise_dataset,batch_size=tw_batch_size,shuffle=True)
     
            # eval model
            twCNN.eval()
            
            s1_tile_res = infer.eval_twcnn(twCNN,test_tw_dl,ps_boundary = tw_boundary,dev=dev,boundary_size=voting_size)
            
            twCNN_params = {'s1_CNNepochs':s1_CNNepochs,
                              'tw_size':tw_size,
                              'vox_spacing':vox_spacing,
                              'thresh_r':thresh_r,
                              'tw_batch_size':tw_batch_size,
                              'tw_lr':tw_lr,
                              'fold':fold,
                              'reading':reading}
            
            for key,value in twCNN_params.items():
                s1_tile_res[key]= [value]*len(s1_tile_res)
            
    
            s1_tile_res.to_csv(os.path.join(save_dir,'csv_inference',twname+'_{}.csv'.format(fold)))
                        