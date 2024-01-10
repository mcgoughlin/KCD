from KCD.Detection.Training import train_utils as tu
import KCD.Detection.Evaluation.eval_scripts as eval_
from KCD.Detection.ModelGenerator import model_generator
from sklearn.model_selection import StratifiedKFold as kfold_strat
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
import warnings
import pandas as pd

def train_cv_slice_model_2d(home = '/Users/mcgoug01/Downloads/Data/',dataname='coreg_ncct',
                         splits:list=[0],folds=5,params:dict=None,train_folds=[0],
                         epochs = None):
    # Suppress warnings
    warnings.filterwarnings("ignore") #makes dgl stop complaining!

    # Initialization
    dev = tu.initialize_device()

    if params==None:params = tu.init_slice2D_params_pretrain()
    else:tu.check_params(params,tu.init_slice2D_params_pretrain())
    model_type = 'TileModel'

    if epochs != None:
        params['epochs'] = epochs

    save_dir = tu.init_training_home(home, dataname)
    slicedataset, test_slicedataset = tu.get_slice_data(home, dataname, params['voxel_size'], params['cancthresh_r_mm'],params['kidthresh_r_mm'],params['depth_z'],params['boundary_z'],params['dilated'],device=dev)
    cases, is_ncct = tu.get_cases(slicedataset)

    # More Initialization
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in splits:
        cv_results = []
        split_path = os.path.join(save_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        split_fp = os.path.join(split_path,'split.npy')
        five_fold_strat = kfold_strat(n_splits=folds, shuffle=True)

        if os.path.exists(split_fp): fold_split = np.load(split_fp,allow_pickle=True)
        else:
            fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))])
            np.save(os.path.join(split_path,split_fp),fold_split)

        for fold,train_index, test_index in fold_split:
            if not (fold in train_folds): continue
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            slice_path = os.path.join(fold_path,model_type)

            if not os.path.exists(fold_path):os.mkdir(fold_path)
            if not os.path.exists(slice_path):os.mkdir(slice_path)

            model = model_generator.return_resnext(size=params['model_size'],dev=dev,in_channels=1,out_channels=3)
            opt = torch.optim.Adam(model.parameters(),lr=params['lr'])

            dl,test_dl = tu.generate_dataloaders(slicedataset,test_slicedataset,cases[train_index],params['batch_size'])
            model = tu.train_model(dl,dev,params['epochs'],loss_fnc,opt,model)
            model_name = '{}_resnext_{}_{}_{}_{}'.format(model_type,params['model_size'],params['epochs'],params['pred_window'],params['lr'])

            if not os.path.exists(os.path.join(slice_path,'model')):
                os.mkdir(os.path.join(slice_path,'model'))
                os.mkdir(os.path.join(slice_path,'csv'))

            torch.save(model,os.path.join(slice_path,'model',model_name))
            model.eval()
            model_res,test_df = eval_.eval_cnn(model,test_dl,dev=dev)
            cv_results.append(test_df)
            model_res.to_csv(os.path.join(slice_path,'csv',model_name+'.csv'))

if __name__ == '__main__':
    import sys
    epochs = int(sys.argv[1])
    fold = int(sys.argv[2])
    dataset = 'kits23_nooverlap'
    home = '/bask/projects/p/phwq4930-renal-canc/KCD_data/Data'
    train_cv_slice_model_2d(home=home,dataname=dataset,splits=[0],train_folds=[fold],epochs=epochs)
