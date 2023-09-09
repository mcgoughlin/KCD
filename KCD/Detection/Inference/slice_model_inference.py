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

def eval_individual_slice_models(home='/Users/mcgoug01/Downloads/Data',trainname='merged_training_set',
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
    inference_dataset = iu.get_slice_data_inference(home,infername,params['voxel_size'],params['fg_thresh'],params['depth_z'],params['boundary_z'],params['dilated'],dev=dev)
    
    model_name = '{}_{}_{}_{}_{}'.format(model_type,params['model_size'],params['epochs'],params['epochs'],params['lr'])
    load_split_path = os.path.join(load_dir,'split_{}'.format(tr_split))    

    boundary = []
    for fold in range(tr_folds):
        fold_path = os.path.join(load_split_path,'fold_{}'.format(fold))
        CNN_path = os.path.join(fold_path,model_type)
        CNN_results = pd.read_csv(os.path.join(CNN_path,'csv',model_name+'.csv'))
        CNN_results = CNN_results[(CNN_results['dataset_loc']=='test')&(CNN_results['Voting Size']==params['pred_window'])]
        boundary.append(CNN_results['Boundary {}'.format(spec_boundary)].values[0])
        print(boundary)
                
    CNN_b =np.median(boundary)
    print('boundary is {}.'.format(CNN_b))
    
    # begin inference!
    foldwise_results = []
    for fold in range(tr_folds):
        fold_path = os.path.join(load_split_path,'fold_{}'.format(fold))
        CNN_path = os.path.join(fold_path,model_type)

        test_dl = DataLoader(inference_dataset,batch_size=params['batch_size'],shuffle=False)                    
        inference_dataset.is_train=False 
        
        CNN = torch.load(os.path.join(CNN_path,'model',model_name),map_location=dev)
        CNN.eval()
        slice_res = iu.eval_cnn(CNN,test_dl,ps_boundary=CNN_b,dev=dev,boundary_size=params['pred_window'])
        foldwise_results.append(slice_res)
        
    all_results = pd.concat(foldwise_results)
    all_results = all_results.groupby(['case','position']).sum().reset_index(level=['case','position'])[['case','position','prediction']]
    all_results['position'] = all_results['position'].apply(lambda x:'right' if x==1 else'left')
    all_results['prediction'] = all_results['prediction'].apply(lambda x:1 if x>=(tr_folds/2) else 0)

    all_results.to_csv(os.path.join(inference_path,'{}_'.format(model_type)+model_name+'.csv'))
    
if __name__ == '__main__':
    home = '/bask/projects/p/phwq4930-renal-canc/KCD_data/Data'
    trainname = 'kits23sncct'
    infername='kits23sncct'
    eval_individual_slice_models(home=home,trainname=trainname,infername=infername)
    
