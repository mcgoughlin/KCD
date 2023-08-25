# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:12:22 2022

@author: mcgoug01
"""
import os

import Ensemble_dataloader as dl_all
import Ensemble_dataloader_shapeonly as dl_shape
import tilewise_dataloader as tw_dl
import torch
import torch.nn as nn
import model_generator
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold as kfold
from sklearn.model_selection import StratifiedKFold as kfold_strat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import dgl

def ROC_func(pred_var,lab,max_pred,intervals=20):
    boundaries = np.arange(-0.01,max_pred+0.01,1/intervals)
    # pred [malig,non-malig]
    sens_spec = []
    is_truely_malig = lab==1
    for boundary in boundaries:
        # pred is zero if benign, 1 if malig
        new_pred = pred_var> boundary
        correct = new_pred == is_truely_malig

        sens = (correct & new_pred).sum()/(is_truely_malig).sum()
        spec = (correct & ~new_pred).sum()/(~is_truely_malig).sum()

        sens_spec.append([sens,spec])
    
    return np.array(sens_spec,dtype=float)

def shape_collate(samples):
    # The input `samples` is a list of pairs
    #  (sv_im,features,graph, obj_label).
    svim,features,graphs, labels = map(list, zip(*samples))

    batched_graph = dgl.batch(graphs)
    return torch.stack(svim),torch.stack(features),batched_graph, torch.stack(labels).squeeze()

def complete_collate(samples):
    # The input `samples` is a list of pairs
    #  (sv_im,features,graph,tile_ims, obj_label).
    svim,features,graphs, tile_ims, labels = map(list, zip(*samples))
    
    # following 2 lines are needed to stack different sized tile stacks into tensor
    # concept: set non-tile stacks to -2, which is the exact value of background in tile images
    max_len = max([len(tiles) for tiles in tile_ims])
    tile_out = torch.ones(len(tile_ims),max_len,*tile_ims[0].shape[2:])*-2 
    
    #slot each tile stack into the tile_out tensor
    for i,tilestack in enumerate(tile_ims):
        tile_out[i,:len(tilestack)] = tilestack[:,0]

    batched_graph = dgl.batch(graphs)
    
    return torch.stack(svim),torch.stack(features),batched_graph, tile_out,torch.stack(labels).squeeze()


if torch.cuda.is_available():
    dev = 'cuda'
else:
    dev = 'cpu'

#### individual model arch. params
tw_size = 'large'
sv_size='small'
gnn_layers = 2
gnn_hiddendim = 10

#### individual model opt. params
object_batchsize = 4
graph_thresh = 1000
mlp_thresh=20000
sv_thresh=8000
sv_lr = 0.0005
mlp_lr = 0.01
gnn_lr = 0.001
# tilewise CNN
vox_spacing = 1
thresh_r = 10
tw_batch_size= 8
tw_lr = 0.0005

#### ensemble model arch. params
ensemble_n1=128
ensemble_n2=16

#### ensemble model opt. params
ensemble_batchsize = 2
combined_threshold = 1000
finetune_lr=0.0002
s1_objepochs = 1000
s1_CNNepochs = 3
s2_epochs = 3

#### data params
tile_dataname = 'coreg_ncct'
shape_ensemble_dataname ='combined_dataset_23andAdds'
complete_ensemble_dataname = 'coreg_ncct'

results=[]  
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/Ensemble_cv_results/tw{}_sv{}'.format(tw_size,
                                                                                                                                                          sv_size)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir,'csv'))
    os.mkdir(os.path.join(save_dir,'png'))

five_fold = kfold(n_splits=5,shuffle=True)
five_fold_strat= kfold_strat(n_splits=5,shuffle=True)
softmax = nn.Softmax(dim=-1)

obj_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/'
tile_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/CNN_dataset'
simpleview_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/CNN_dataset/SimpleView'

shapedataset = dl_shape.EnsembleDataset(obj_path,
                         simpleview_path,
                         data_name=shape_ensemble_dataname,
                         graph_thresh=graph_thresh,mlp_thresh=mlp_thresh,sv_thresh=sv_thresh)

test_shapedataset = dl_shape.EnsembleDataset(obj_path,
                         simpleview_path,
                         data_name=shape_ensemble_dataname,
                         graph_thresh=0,mlp_thresh=0,sv_thresh=0)

tilewise_dataset = tw_dl.get_dataset(tile_path,voxel_spacing=vox_spacing,thresh_r_mm=thresh_r,data_name=tile_dataname,is_masked=True)

finaldataset = dl_all.EnsembleDataset(obj_path,
                                      tile_path,
                                      simpleview_path,
                                      data_name=complete_ensemble_dataname,thresh=combined_threshold)

test_finaldataset = dl_all.EnsembleDataset(obj_path,
                                           tile_path,
                                            simpleview_path,
                                            data_name=complete_ensemble_dataname,thresh=0)

# highlight cases that are authentic ncct as 1, others as 0
# this allows the k-fold case allocation to evenly split along the authentic ncct class
cases = np.unique(shapedataset.cases.values)
is_ncct = np.array([0 if case.startswith('case') else 1 for case in cases])

test_res,train_res = [], []

for reading in [0,1,2]:
    fold_split = [(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))]
    for fold,train_index, test_index in fold_split: 
        MLP = model_generator.return_MLP(dev=dev)
        GNN = model_generator.return_graphnn(num_features=4,num_labels=2,layers_deep=gnn_layers,hidden_dim=gnn_hiddendim,dev=dev)
        CNN = model_generator.return_efficientnet(dev=dev,in_channels=6,out_channels=2)
        twCNN = model_generator.return_efficientnet(size=tw_size,dev=dev,in_channels=1,out_channels=3)
        
        print("\nFold {} training.".format(fold))
        loss_fnc = nn.CrossEntropyLoss().to(dev)
        
        ######## Individual Object Model Training ########
        print("Training object classifiers")
        
        CNNopt = torch.optim.Adam(CNN.parameters(),lr=sv_lr)
        GNNopt = torch.optim.Adam(GNN.parameters(),lr=gnn_lr)
        MLPopt = torch.optim.Adam(MLP.parameters(),lr=mlp_lr)
    
        shapedataset.apply_foldsplit(train_cases = cases[train_index])
        test_shapedataset.apply_foldsplit(train_cases = cases[train_index])
        
        fold_cases = np.array([case for case in cases[train_index] if not case.startswith('case')])
        fold_cases = np.array([case if not case.startswith('KiTS') else case.replace('-','_') for case in fold_cases ])

        finaldataset.apply_foldsplit(train_cases = fold_cases)
        test_finaldataset.apply_foldsplit(train_cases = fold_cases)
        test_shapedataset.is_train=True #test accuracy on training data and test data. training data first, so is_train = true
        shapedataset.is_train=True
        test_finaldataset.is_train=True #test accuracy on training data and test data. training data first, so is_train = true
        finaldataset.is_train=True
        
        dl = DataLoader(shapedataset,batch_size=object_batchsize,shuffle=True,collate_fn=shape_collate)
        test_dl = DataLoader(test_shapedataset,batch_size=object_batchsize,shuffle=True,collate_fn=shape_collate)
        
        complete_dl = DataLoader(finaldataset,batch_size=ensemble_batchsize,shuffle=True,collate_fn=complete_collate)
        test_complete_dl = DataLoader(test_finaldataset,batch_size=ensemble_batchsize,shuffle=True,collate_fn=complete_collate)
        
        for i in range(s1_objepochs):
            print("\nEpoch {}".format(i))
            for sv_im,features,graph, lb in dl:
                sv_lb,feat_lb,graph_lb = lb.T
                CNNpred = CNN(sv_im.to(dev))
                MLPpred = MLP(features.to(dev))
                GNNpred = GNN(graph.to(dev))
                for pred,opt,label in zip([CNNpred,MLPpred,GNNpred],[CNNopt,MLPopt,GNNopt], [sv_lb,feat_lb,graph_lb]):
                    loss = loss_fnc(pred,label.to(dev))
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
                break
            break
        
        
        ######## Individual CNN Model Training ########
        
        print("Training tile classifier")

        
        tilewise_dataset.apply_foldsplit(train_cases = fold_cases)
        tilewise_dataset.is_train=True
        TWopt = torch.optim.Adam(twCNN.parameters(),lr=tw_lr)
        
        tw_dl = DataLoader(tilewise_dataset,batch_size=tw_batch_size,shuffle=True)
        
        for i in range(s1_CNNepochs):
            print("Epoch {}\n".format(i))
            for x,lb in tw_dl:
                opt.zero_grad()
                pred = twCNN(x.to(dev))
                loss = loss_fnc(pred,lb.to(dev))
                loss.backward()
                opt.step()
                
                break
            break
        
        
        ######## Shape-Ensemble Model Training ########
        
        print("Training object classifier ensemble")
        
        shape_ensemble = model_generator.return_shapeensemble(CNN,MLP,GNN,
                                                              n1=ensemble_n1,
                                                              n2=ensemble_n2)
        
        SEopt = torch.optim.Adam(shape_ensemble.parameters(),lr=finetune_lr)
        dl.dataset.graph_thresh = combined_threshold
        dl.dataset.mlp_thresh = combined_threshold
        dl.dataset.sv_thresh = combined_threshold

        # train models ensembled
        for i in range(s2_epochs):
            print("\nEpoch {}".format(i))
            for sv_im,features,graph, lb in dl:
                label,_,_ = lb.T
                SEopt.zero_grad()
                SEpred = shape_ensemble(features,sv_im,graph)
                loss = loss_fnc(SEpred,label.to(dev))
                loss.backward()
                SEopt.step()
                break
            break
        
        ######## Complete Ensemble Model Training ########
        
        print("Training all ensemble")

        
        complete_ensemble = model_generator.return_completeensemble(shape_ensemble,
                                                                    twCNN,
                                                                  n1=ensemble_n1,
                                                                  n2=ensemble_n2)
        
        CEopt = torch.optim.Adam(complete_ensemble.parameters(),lr=finetune_lr)
        
        # train models ensembled
        for i in range(s2_epochs):
            print("\nEpoch {}".format(i))
            for sv_im,features,graph,tile_ims, lb in complete_dl:
                # assert(1==2)
                CEopt.zero_grad()
                CEpred = complete_ensemble(features,sv_im,graph,tile_ims)
                loss = loss_fnc(CEpred,lb.to(dev))
                loss.backward()
                CEopt.step()
                print(CEpred)
                break
            break
        
        assert(1==2)
    
        
        