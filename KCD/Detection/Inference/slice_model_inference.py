# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:12:22 2022

@author: mcgoug01
"""
from KCD.Detection.Inference import infer_utils as iu

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import dgl
import warnings
import pandas as pd

def eval_individual_shape_models(home='/Users/mcgoug01/Downloads/Data',trainname='merged_training_set',
                                 infername='add_ncct_unseen',params:dict=None,tr_split=0,tr_folds=5,
                                 spec_boundary=98,is_3D=True):
    warnings.filterwarnings("ignore")
    if is_3D:
        if params==None:params = iu.init_slice3D_params()
        else:iu.check_params(params,iu.init_shape3D_params())
        model_type = 'PatchModel'
    else:
        model_type = 'TileModel'
        if params==None:params = iu.init_slice2D_params()
        else:iu.check_params(params,iu.init_shape2D_params())
    
    inference_path = os.path.join(home,'inference')
    inference_path = iu.init_inference_home(inference_path,infername,trainname,tr_split)
    
    #### path init  
    load_dir = iu.init_training_home(home,trainname)
    
    dev = iu.initialize_device()

    #### init dataset
    inference_dataset = iu.get_slice_data_inference(home,infername,dev=dev)
    
    model_name = '{}_{}_{}_{}_{}'.format(model_type,params['model_size'],params['epochs'],params['epochs'],params['lr'])
    split_path = os.path.join(load_dir,'split_{}'.format(tr_split))

    boundary = []
    for fold in range(tr_folds):
        fold_path = os.path.join(split_path,'fold_{}'.format(fold))
        CNN_path = os.path.join(fold_path,model_type)
        CNN_results = pd.read_csv(os.path.join(CNN_path,'csv',model_name+'.csv'))
        CNN_results = CNN_results[(CNN_results['dataset_loc']=='test')&(CNN_results['Voting Size']==params['voting_size'])]
        boundary.append(CNN_results['Boundary {}'.format(spec_boundary)].values[0])
        print(boundary)
                
    CNN_b =np.median(boundary)
    print('boundary is {}.'.format(CNN_b))
    
    # Here, we decide the threshold boundary of our models, based on cross-validation data
    load_split_path = os.path.join(load_dir,'split_{}'.format(tr_split))    
    MLP_boundary,GNN_boundary = [],[]
    for fold in range(tr_folds):
        MLP_name = '{}_{}_{}_{}'.format(params['s1_objepochs'],params['mlp_thresh'],params['mlp_lr'],params['object_batchsize'])
        GNN_name = '{}_{}_{}_{}_{}_{}_{}'.format(params['s1_objepochs'],params['graph_thresh'],params['gnn_lr'],params['gnn_layers'],params['gnn_hiddendim'],params['gnn_neighbours'],params['object_batchsize'])
        fold_path = os.path.join(load_split_path,'fold_{}'.format(fold))
        
        for name,filename,boundary in zip(['MLP','GNN'],[MLP_name,GNN_name],[MLP_boundary,GNN_boundary]):
            shape_path = os.path.join(fold_path,name)
            results = pd.read_csv(os.path.join(shape_path,'csv',filename+'.csv'))
            results = results[(results['dataset_loc']=='test')]
            boundary.append(results['Boundary {}'.format(spec_boundary)].values[0])
    MLP_b,GNN_b =np.median(MLP_boundary),np.median(GNN_boundary)
    
    # begin inference!
    foldwise_results = []
    for fold in range(tr_folds):
        MLP_name = '{}_{}_{}_{}'.format(params['s1_objepochs'],params['mlp_thresh'],params['mlp_lr'],params['object_batchsize'])
        GNN_name = '{}_{}_{}_{}_{}_{}_{}'.format(params['s1_objepochs'],params['graph_thresh'],params['gnn_lr'],params['gnn_layers'],params['gnn_hiddendim'],params['gnn_neighbours'],params['object_batchsize'])
        fold_path = os.path.join(load_split_path,'fold_{}'.format(fold))
    
        test_dl = DataLoader(inference_dataset,batch_size=params['object_batchsize'],shuffle=False,collate_fn=iu.shape_collate_unlabelled)                    
        inference_dataset.is_train=False 
        
        MLP = torch.load(os.path.join(fold_path,'MLP','model',MLP_name),map_location=dev)
        GNN = torch.load(os.path.join(fold_path,'GNN','model',GNN_name),map_location=dev)

        MLP.eval(),GNN.eval()
        s2_shape_res = iu.eval_shape_individual_models(MLP,GNN,test_dl,MLP_boundary=MLP_b,GNN_boundary=GNN_b,dev=dev)
        foldwise_results.append(s2_shape_res)
        
    all_results = pd.concat(foldwise_results)
    all_results = all_results.groupby(['case','position']).sum().reset_index(level=['case','position'])[['case','position','MLPpred-hard','GNNpred-hard']]
    all_results['position'] = all_results['position'].apply(lambda x:'right' if x==1 else'left')
    all_results['MLPpred-hard'] = all_results['MLPpred-hard'].apply(lambda x:1 if x>=(tr_folds/2) else 0)
    all_results['GNNpred-hard'] = all_results['GNNpred-hard'].apply(lambda x:1 if x>=(tr_folds/2) else 0)
    
    MLP_res = all_results[['case','position','MLPpred-hard']]
    GNN_res = all_results[['case','position','GNNpred-hard']]

    MLP_res.to_csv(os.path.join(inference_path,'MLP_'+MLP_name+'.csv'))
    GNN_res.to_csv(os.path.join(inference_path,'GNN_'+GNN_name+'.csv'))
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir,'csv'))
    os.mkdir(os.path.join(save_dir,'png'))


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
                        
     