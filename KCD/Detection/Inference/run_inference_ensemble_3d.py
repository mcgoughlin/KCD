# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:12:22 2022

@author: mcgoug01
"""
import Ensemble_dataloader_shapeonly_infer as dl_shape
import tilewise_dataloader_3D as tw_dl
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
tw_lr =  0.001
tw_size='small'
voting_size=1

#### ensemble model opt. params
s1_CNNepochs = 30

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

test_tilewise_dataset = tw_dl.get_dataset(tile_path,voxel_spacing=vox_spacing,thresh_r_mm=0,data_name=tile_dataname,is_masked=True,depth=20)


# highlight cases that are authentic ncct as 1, others as 0
# this allows the k-fold case allocation to evenly split along the authentic ncct class
cases = np.unique(test_shapedataset.cases.values)
twname = 'resnext3D_{}_{}_{}_{}'.format(tw_size,s1_CNNepochs,tw_batch_size,tw_lr)
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
            twCNN_path = os.path.join(fold_path,'twCNN_3D')
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
            twCNN_path = os.path.join(fold_path,'twCNN_3D')
            
            

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
                        
            # ######## Shape-Ensemble Model Training ########
            # test_shapedataset.is_train=False #test accuracy on training data and test data. training data first, so is_train = true
            # print("Training object classifier ensemble")
            # name = 'ShapeEnsemble_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(ensemble_n1,ensemble_n2,fold,shape_freeze_epochs,shape_unfreeze_epochs,
            #                                                                sv_thresh,graph_thresh,mlp_thresh,s1_svcnnepochs,s1_gnnepochs,s1_mlpepochs)
    
            # shape_ensemble_fp = os.path.join(save_dir,'shape_ensemble',name)
            # shape_ensemble = torch.load(shape_ensemble_fp)
            
            # shape_cv_results = pd.read_csv(os.path.join(save_dir,'csv_trainedseperate_V5',name+'_{}.csv'.format(reading)))
            # shape_boundary = shape_cv_results['Boundary 98'].values[0]/1020
                         
            # shape_ensemble.eval()
            # s2_shape_res = infer.eval_shape_ensemble(shape_ensemble,test_dl,boundary=shape_boundary,
            #                                          dev=dev)
            
            # s2_shape_params = {'s2_spec_multiplier':s2_spec_multiplier,
            #                    'shape_freeze_epochs':shape_freeze_epochs,
            #                    'shape_freeze_lr':shape_freeze_lr,
            #                    'shape_unfreeze_epochs':shape_unfreeze_epochs,
            #                    'shape_unfreeze_lr':shape_unfreeze_lr,
            #                    'ensemble_n1':ensemble_n1,
            #                    'ensemble_n2':ensemble_n2}
            
            # s2_shape_params.update(indobj_params)
            
            
                                                                           
            # s2_shape_res.to_csv(os.path.join(save_dir,'csv_inference',name+'_{}.csv'.format(reading)))
                        
     