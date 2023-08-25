# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:12:22 2022

@author: mcgoug01
"""
import Ensemble_dataloader_shapeonly_nocnn_infer as dl_shape
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
    features,graphs, labels = map(list, zip(*samples))

    batched_graph = dgl.batch(graphs)
    return torch.stack(features).to(dev),batched_graph.to(dev), torch.stack(labels).squeeze(dim=0).to(dev)

def compute_loss_weighting(spec_multiplier,num_classes=2):
    sens = num_classes/(spec_multiplier+1)
    spec = spec_multiplier*sens
    
    return sens,spec

gnn_layers = 5
gnn_hiddendim = 25
gnn_neighbours = 2
ensemble_n1 = 128
ensemble_n2 = 32

#### individual model opt. params
object_batchsize = 8
graph_thresh = 500
mlp_thresh=20000
mlp_lr = 0.01
gnn_lr = 1e-3


#### ensemble model opt. params
s1_objepochs = 100
shape_unfreeze_epochs = 2
shape_freeze_epochs = 25
shape_freeze_lr = 1e-3
shape_unfreeze_lr = 1e-3
combined_threshold = 500


#### sens/spec weighting during training
s1_spec_multiplier = 1.0
s1_spec, s1_sens = compute_loss_weighting(s1_spec_multiplier)


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
                         data_name=shape_ensemble_dataname,
                         graph_thresh=0,mlp_thresh=0,dev=dev)


# highlight cases that are authentic ncct as 1, others as 0
# this allows the k-fold case allocation to evenly split along the authentic ncct class
cases = np.unique(test_shapedataset.cases.values)
boundary = []

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
            name = 'ShapeEnsemble_noCNN_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(ensemble_n1,ensemble_n2,fold,shape_freeze_epochs,shape_unfreeze_epochs
                                                                              ,graph_thresh,mlp_thresh,s1_objepochs,shape_freeze_lr,shape_unfreeze_lr)
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            shape_path = os.path.join(fold_path,'ShapeEnsemble')
            results = pd.read_csv(os.path.join(shape_path,'csv',name+'.csv'))
            results = results[(results['dataset_loc']=='test')]
            boundary.append(results['Boundary 98'].values[0])
            print(boundary)
            
boundary =np.median(boundary)
print('boundary is {}.'.format(boundary))

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
            MLP_path = os.path.join(fold_path,'MLP')
            twCNN_path = os.path.join(fold_path,'twCNN')
            GNN_path = os.path.join(fold_path,'GNN')
            Shape_path = os.path.join(fold_path,'ShapeEnsemble')
            
            paths = [MLP_path,Shape_path,twCNN_path,GNN_path]
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
                
            for path in paths:
                if not os.path.exists(path):
                    os.mkdir(path)
                    
                modpath = os.path.join(path,'model')
                csvpath = os.path.join(path,'csv')
                
                if not os.path.exists(modpath):
                    os.mkdir(modpath)
                    
                if not os.path.exists(csvpath):
                    os.mkdir(csvpath)
                    
            name = 'ShapeEnsemble_noCNN_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(ensemble_n1,ensemble_n2,fold,shape_freeze_epochs,shape_unfreeze_epochs
                                                                              ,graph_thresh,mlp_thresh,s1_objepochs,shape_freeze_lr,shape_unfreeze_lr)
            

            print("\nFold {} training.".format(fold))
            
            ######## Individual Object Model Eval ########
        
            test_shapedataset.apply_foldsplit(train_cases = cases)
            test_dl = DataLoader(test_shapedataset,batch_size=object_batchsize,shuffle=True,collate_fn=shape_collate)
            print('cases',test_dl.dataset.train_cases)
                        

            ######## Shape-Ensemble Model Training ########
            test_shapedataset.is_train=False #test accuracy on training data and test data. training data first, so is_train = true
            print("Training object classifier ensemble")

            shape_ensemble_fp = os.path.join(load_dir,'split_0','fold_{}'.format(fold),'ShapeEnsemble','model',name)
            shape_ensemble = torch.load(shape_ensemble_fp,map_location=dev)
            shape_ensemble.device = dev
            shape_ensemble.GNN.device = dev
            # shape_ensemble.MLP = shape_ensemble.MLP.to(dev)
            # shape_ensemble.GNN = shape_ensemble.GNN.to(dev)
            # shape_ensemble.process1 = shape_ensemble.process1.to(dev)
            # shape_ensemble.final = shape_ensemble.final.to(dev)
            shape_ensemble.eval(),shape_ensemble.MLP.eval(),shape_ensemble.GNN.eval()
            s2_shape_res = infer.eval_shape_ensemble_nocnn(shape_ensemble,test_dl,boundary=boundary,
                                                      dev=dev)
            
                                                                           
            s2_shape_res.to_csv(os.path.join(save_dir,'csv_shape_nocnn',name+'.csv'))
                        
     