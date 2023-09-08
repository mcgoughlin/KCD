# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:12:22 2022

@author: mcgoug01
"""
import KCD.Detection.Inference.infer_utils as iu
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import warnings
import pandas as pd



#### data params
shape_ensemble_dataname ='add_ncct_unseen'

#### ignore all warnings - due to dgl being very annoying

def eval_individual_shape_models(home='/Users/mcgoug01/Downloads/Data',trainname='merged_training_set',
                                 infername='add_ncct_unseen',params:dict=None,tr_split=0,tr_folds=5):
    warnings.filterwarnings("ignore")
    if params==None:params = iu.init_shape_params()
    else:iu.check_params(params,iu.init_shape_params())
    
    inference_path = os.path.join(home,'inference')
    iu.create_directory(inference_path),iu.create_directory(os.path.join(inference_path,'csv')),iu.create_directory(os.path.join(inference_path,'png'))
    
    #### path init  
    load_dir = iu.init_training_home(home,trainname)
    
    dev = iu.initialize_device()

    #### init dataset
    inference_dataset = iu.get_shape_data_inference(home,infername,dev=dev)
    cases = np.unique(inference_dataset.cases.values)

    split_path = os.path.join(load_dir,'split_{}'.format(tr_split))
    split_fp = os.path.join(split_path,'split.npy')
    fold_split = np.load(split_fp,allow_pickle=True)
    
    MLP_boundary,GNN_boundary = [],[] # Here, we decide the threshold boundary of our models, based on cross-validation data
    for fold in range(tr_folds):
        MLP_name = '{}_{}_{}_{}'.format(params['s1_objepochs'],params['mlp_thresh'],params['mlp_lr'],params['object_batchsize'])
        GNN_name = '{}_{}_{}_{}_{}_{}_{}'.format(params['s1_objepochs'],params['graph_thresh'],params['gnn_lr'],params['gnn_layers'],params['gnn_hiddendim'],params['gnn_neighbours'],params['object_batchsize'])
        fold_path = os.path.join(split_path,'fold_{}'.format(fold))
        
        for name,filename,boundary in zip(['MLP','GNN'],[MLP_name,GNN_name],[MLP_boundary,GNN_boundary]):
            shape_path = os.path.join(fold_path,name)
            results = pd.read_csv(os.path.join(shape_path,'csv',filename+'.csv'))
            results = results[(results['dataset_loc']=='test')]
            boundary.append(results['Boundary 98'].values[0])
                
    MLP_b,GNN_b =np.median(MLP_boundary),np.median(GNN_boundary)
    print('boundaries are {} and {}.'.format(MLP_b,GNN_b))
    assert(1==2)
    # begin inference!
    for fold in range(tr_folds):
        MLP_name = '{}_{}_{}_{}'.format(params['s1_objepochs'],params['mlp_thresh'],params['mlp_lr'],params['object_batchsize'])
        GNN_name = '{}_{}_{}_{}_{}_{}_{}'.format(params['s1_objepochs'],params['graph_thresh'],params['gnn_lr'],params['gnn_layers'],params['gnn_hiddendim'],params['gnn_neighbours'],params['object_batchsize'])
        fold_path = os.path.join(split_path,'fold_{}'.format(fold))
    
        inference_dataset.apply_foldsplit(train_cases = cases)
        test_dl = DataLoader(inference_dataset,batch_size=params['object_batchsize'],shuffle=False,collate_fn=iu.shape_collate)                    
        inference_dataset.is_train=False 

        shape_ensemble = torch.load(os.path.join(),map_location=dev)


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
                   
if __name__ == '__main__':
    trainname='coreg_ncct'
    eval_individual_shape_models(trainname=trainname)
         