# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:12:22 2022

@author: mcgoug01
"""
import KCD.Detection.Inference.infer_utils as iu
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
import pandas as pd


def eval_individual_shape_models(home='/Users/mcgoug01/Downloads/Data',trainname='merged_training_set',
                                 infername='add_ncct_unseen',params:dict=None,tr_split=0,tr_folds=5,spec_boundary=98):
    warnings.filterwarnings("ignore")
    if params==None:params = iu.init_shape_params()
    else:iu.check_params(params,iu.init_shape_params())
    
    inference_path = os.path.join(home,'inference')
    inference_path = iu.init_inference_home(inference_path,infername,trainname,tr_split)
    
    #### path init  
    load_dir = iu.init_training_home(home,trainname)
    
    dev = iu.initialize_device()

    #### init dataset
    inference_dataset = iu.get_shape_data_inference(home,infername,dev=dev)
    
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
        shape_res = iu.eval_shape_individual_models(MLP,GNN,test_dl,MLP_boundary=MLP_b,GNN_boundary=GNN_b,dev=dev)
        foldwise_results.append(shape_res)
        
    all_results = pd.concat(foldwise_results)
    all_results = all_results.groupby(['case','position']).sum().reset_index(level=['case','position'])[['case','position','MLPpred-hard','GNNpred-hard']]
    all_results['position'] = all_results['position'].apply(lambda x:'right' if x==1 else'left')
    all_results['MLPpred-hard'] = all_results['MLPpred-hard'].apply(lambda x:1 if x>=(tr_folds/2) else 0)
    all_results['GNNpred-hard'] = all_results['GNNpred-hard'].apply(lambda x:1 if x>=(tr_folds/2) else 0)
    
    MLP_res = all_results[['case','position','MLPpred-hard']]
    GNN_res = all_results[['case','position','GNNpred-hard']]

    MLP_res.to_csv(os.path.join(inference_path,'MLP_'+MLP_name+'.csv'))
    GNN_res.to_csv(os.path.join(inference_path,'GNN_'+GNN_name+'.csv'))
    
    
def eval_shape_ensemble(home='/Users/mcgoug01/Downloads/Data',trainname='merged_training_set',
                                 infername='add_ncct_unseen',params:dict=None,tr_split=0,tr_folds=5,spec_boundary=98):
    warnings.filterwarnings("ignore")
    if params==None:params = iu.init_shape_params()
    else:iu.check_params(params,iu.init_shape_params())
    
    inference_path = os.path.join(home,'inference')
    inference_path = iu.init_inference_home(inference_path,infername,trainname,tr_split)
    
    #### path init  
    load_dir = iu.init_training_home(home,trainname)
    
    dev = iu.initialize_device()

    #### init dataset
    inference_dataset = iu.get_shape_data_inference(home,infername,dev=dev)
    
    # Here, we decide the threshold boundary of our models, based on cross-validation data
    load_split_path = os.path.join(load_dir,'split_{}'.format(tr_split))    
    boundary = []
    for fold in range(tr_folds):
        ensemble_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(params['ensemble_n1'],params['ensemble_n2'],fold,params['shape_freeze_epochs'],params['shape_unfreeze_epochs'],params['graph_thresh'],params['mlp_thresh'],params['s1_objepochs'],params['shape_freeze_lr'],params['shape_unfreeze_lr'])
        fold_path = os.path.join(load_split_path,'fold_{}'.format(fold))
        shape_path = os.path.join(fold_path,'shape_ensemble')
        results = pd.read_csv(os.path.join(shape_path,'csv',ensemble_name+'.csv'))
        results = results[(results['dataset_loc']=='test')]
        boundary.append(results['Boundary {}'.format(spec_boundary)].values[0])
    ensemble_b = np.median(boundary)
    # begin inference!
    foldwise_results = []
    for fold in range(tr_folds):
        ensemble_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(params['ensemble_n1'],params['ensemble_n2'],params['shape_freeze_epochs'],params['shape_unfreeze_epochs'],params['graph_thresh'],params['mlp_thresh'],params['s1_objepochs'],params['shape_freeze_lr'],params['shape_unfreeze_lr'])
        fold_path = os.path.join(load_split_path,'fold_{}'.format(fold))
    
        test_dl = DataLoader(inference_dataset,batch_size=params['object_batchsize'],shuffle=False,collate_fn=iu.shape_collate_unlabelled)                    
        inference_dataset.is_train=False 
        
        ensemble = torch.load(os.path.join(fold_path,'shape_ensemble','model',ensemble_name),map_location=dev)

        ensemble.eval()
        s2_shape_res = iu.eval_shape_ensemble(ensemble,test_dl,boundary=ensemble_b,dev=dev)
        foldwise_results.append(s2_shape_res)
        
    all_results = pd.concat(foldwise_results)
    all_results = all_results.groupby(['case','position']).sum().reset_index(level=['case','position'])[['case','position','Ensemblepred-hard']]
    all_results['Ensemblepred-hard'] = all_results['Ensemblepred-hard'].apply(lambda x:1 if x>=(tr_folds/2) else 0)
    all_results.to_csv(os.path.join(inference_path,'ensemble_'+ensemble_name+'.csv'))

                   
if __name__ == '__main__':
    trainname='merged_training_set'
    eval_individual_shape_models(trainname=trainname)
    eval_shape_ensemble(trainname=trainname)
         