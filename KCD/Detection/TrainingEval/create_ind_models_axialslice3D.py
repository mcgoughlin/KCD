# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:12:22 2022

@author: mcgoug01
"""
import Ensemble_dataloader_shapeonly as dl_shape
import tilewise_dataloader_3D as tw_dl
import eval_scripts as eval_
import os
import torch
import torch.nn as nn
import classifier_model_generator as model_generator
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold as kfold_strat
import numpy as np
import sys
import dgl
import warnings
import gc

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

fold = int(sys.argv[1])
vox_spacing = 1
thresh_r =10
tw_batch_size= 32
tw_lr = 0.001
depth = 20

#### ensemble model opt. params
s1_CNNepochs = 35

print('Num s1 epochs:',s1_CNNepochs)
#### sens/spec weighting during training
#### data params
tile_dataname = 'coreg_ncct'
shape_ensemble_dataname ='combined_dataset_23andAdds'

#### ignore all warnings - due to dgl being very annoying

ignore = True
if ignore: warnings.filterwarnings("ignore")

#### path init 
 
save_dir = '/bask/projects/p/phwq4930-renal-canc/EnsembleResults/saved_info'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)



obj_path = '/bask/projects/p/phwq4930-renal-canc/GraphData/'
tile_path = '/bask/projects/p/phwq4930-renal-canc/'
simpleview_path = '/bask/projects/p/phwq4930-renal-canc/SimpleView'

#### training housekeeping
results=[] 
five_fold_strat= kfold_strat(n_splits=5,shuffle=True)
softmax = nn.Softmax(dim=-1)
loss_fnc = nn.CrossEntropyLoss().to(dev)

#### init all datasets needed 

shapedataset = dl_shape.EnsembleDataset(obj_path,
                         simpleview_path,
                         data_name=shape_ensemble_dataname,
                         graph_thresh=0,mlp_thresh=0,sv_thresh=0)

tilewise_dataset = tw_dl.get_dataset(tile_path,voxel_spacing=vox_spacing,thresh_r_mm=thresh_r,data_name=tile_dataname,depth=depth,is_masked=True)
test_tilewise_dataset = tw_dl.get_dataset(tile_path,voxel_spacing=vox_spacing,thresh_r_mm=0,data_name=tile_dataname,depth=depth,is_masked=True)

# highlight cases that are authentic ncct as 1, others as 0
# this allows the k-fold case allocation to evenly split along the authentic ncct class
cases = np.unique(shapedataset.cases.values)
is_ncct = np.array([0 if case.startswith('case') else 1 for case in cases])

test_res,train_res = [], []
for tw_size in ['small']:
    for reading in range(1):
        for split in range(3):
            split_path = os.path.join(save_dir,'split_{}'.format(split))
            if not os.path.exists(split_path):
                os.mkdir(split_path)
                
            split_fp = os.path.join(split_path,'split.npy')
            if os.path.exists(split_fp):
                fold_split = np.load(split_fp,allow_pickle=True)
            else:  
                fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))])
                np.save(os.path.join(split_path,split_fp),fold_split)
                
            # begin training!
            fold, train_index, test_indexc = fold_split[fold]
            # for fold,train_index, test_index in fold_split: 
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            MLP_path = os.path.join(fold_path,'MLP')
            svCNN_path = os.path.join(fold_path,'svCNN')
            twCNN3d_path = os.path.join(fold_path,'twCNN_3D')
            twCNN_path = os.path.join(fold_path,'twCNN')
            GNN_path = os.path.join(fold_path,'GNN')
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
            
            paths = [MLP_path,svCNN_path,twCNN_path,GNN_path,twCNN3d_path]
            
            for path in paths:
                if not os.path.exists(path):
                    os.mkdir(path)
                    
                modpath = os.path.join(path,'model')
                csvpath = os.path.join(path,'csv')
                
                if not os.path.exists(modpath):
                    os.mkdir(modpath)
                    
                if not os.path.exists(csvpath):
                    os.mkdir(csvpath)
                
                
            # extract ncct cases from the fold split, 
            # use this later to fine-tune only on authentic ncct.
            ncct_cases = np.array([case for case in cases[train_index] if not case.startswith('case')])
            ncct_cases = np.array([case if not case.startswith('KiTS') else case.replace('-','_') for case in ncct_cases ])
        
            # init models
            twCNN = model_generator.return_resnext3D(size=tw_size,dev=dev,in_channels=1,out_channels=3)
            
        
            print("\nFold {} training.".format(fold))
    
            ######## Individual CNN Model Training ########
            print("Training tile classifier")
            
            twCNN.train()
            
            test_tilewise_dataset.apply_foldsplit(train_cases = ncct_cases)
            tilewise_dataset.apply_foldsplit(train_cases = ncct_cases)
            
            test_tilewise_dataset.is_train=True
            tilewise_dataset.is_train=True
            
            TWopt = torch.optim.Adam(twCNN.parameters(),lr=tw_lr)
            
            tw_dl = DataLoader(tilewise_dataset,batch_size=tw_batch_size,shuffle=True)
            test_tw_dl = DataLoader(test_tilewise_dataset,batch_size=tw_batch_size,shuffle=True)
            loss_fnc.weight = None
            for i in range(s1_CNNepochs):
                print("Epoch {}\n".format(i))
                for x,lb in tw_dl:
                    TWopt.zero_grad()
                    pred = twCNN(x.to(dev))
                    loss = loss_fnc(pred,lb.to(dev))
                    loss.backward()
                    TWopt.step()
                    
            twCNN.eval()
                    
            twname = 'resnext3D_{}_{}_{}_{}'.format(tw_size,s1_CNNepochs,tw_batch_size,tw_lr)
            
            torch.save(twCNN,os.path.join(twCNN3d_path,'model',twname))
            s1_tile_res = eval_.eval_twcnn(twCNN,test_tw_dl,dev=dev)
            s1_tile_res.to_csv(os.path.join(twCNN3d_path,'csv',twname+'_{}.csv'.format(reading)))

                    