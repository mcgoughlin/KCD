import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(2)
np.random.seed(2)

from sklearn.model_selection import StratifiedKFold as kfold_strat
import KCD.Detection.Evaluation.eval_scripts as eval_
from KCD.Detection.ModelGenerator import model_generator
from KCD.Detection.Training import train_utils as tu


def train_cv_individual_models(home = '/media/mcgoug01/nvme/SecondYear/Data/',dataname='merged_training_set',
                            splits:list=[0],folds=5,params:dict=None):
    # Suppress warnings
    warnings.filterwarnings("ignore") #makes dgl stop complaining!

    # Initialization
    dev = tu.initialize_device()
    
    if params==None:params = tu.init_shape_params()
    else:tu.check_params(params,tu.init_shape_params())
    
    save_dir = tu.init_training_home(home, dataname)
    shapedataset, test_shapedataset = tu.get_shape_data(home, dataname, params['graph_thresh'], params['mlp_thresh'],ensemble=False,dev=dev)
    cases, is_ncct = tu.get_cases(shapedataset)

    # More Initialization
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in splits:
        cv_results = []
        split_path = os.path.join(save_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)
            
        split_fp = os.path.join(split_path,'split.npy')
        five_fold_strat = kfold_strat(n_splits=folds, shuffle=True)

        if os.path.exists(split_fp):
            fold_split = np.load(split_fp,allow_pickle=True)
        else:  
            fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))],dtype=object)
            np.save(os.path.join(split_path,split_fp),fold_split)


        for fold,train_index, test_index in fold_split:
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            MLP_path = os.path.join(fold_path,'MLP')
            GNN_path = os.path.join(fold_path,'GNN')
            
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
                os.mkdir(MLP_path)
                os.mkdir(GNN_path)
                
            MLP = model_generator.return_MLP(dev=dev)
            GNN = model_generator.return_GNN(num_features=4,num_labels=2,layers_deep=params['gnn_layers'],hidden_dim=params['gnn_hiddendim'],neighbours=params['gnn_neighbours'],dev=dev)
            GNNopt = torch.optim.Adam(GNN.parameters(),lr=params['gnn_lr'])
            MLPopt = torch.optim.Adam(MLP.parameters(),lr=params['mlp_lr'])

            dl,test_dl = tu.generate_dataloaders(shapedataset,test_shapedataset,cases[train_index],params['object_batchsize'],tu.shape_collate)
            MLP,GNN = tu.train_shape_models(dl,dev,params['s1_objepochs'],loss_fnc,MLPopt,GNNopt,MLP,GNN)

            MLP_name = '{}_{}_{}_{}'.format(params['s1_objepochs'],params['mlp_thresh'],params['mlp_lr'],params['object_batchsize'])
            GNN_name = '{}_{}_{}_{}_{}_{}_{}'.format(params['s1_objepochs'],params['graph_thresh'],params['gnn_lr'],params['gnn_layers'],params['gnn_hiddendim'],params['gnn_neighbours'],params['object_batchsize'])
            for modpath in [MLP_path,GNN_path]:
                if not os.path.exists(os.path.join(modpath,'model')):os.mkdir(os.path.join(modpath,'model'))
                if not os.path.exists(os.path.join(modpath,'csv')):os.mkdir(os.path.join(modpath,'csv'))

            torch.save(MLP,os.path.join(MLP_path,'model',MLP_name))
            torch.save(GNN,os.path.join(GNN_path,'model',GNN_name))
            
            GNN.eval(),MLP.eval()
            shape_model_res,test_df = eval_.eval_shape_models(GNN,MLP,test_dl,dev=dev)
            shape_model_res = shape_model_res.reset_index(level=['model'])
            cv_results.append(test_df)
            
            GNN_res = shape_model_res[shape_model_res['model']=='GNN'].drop('model',axis=1)
            MLP_res = shape_model_res[shape_model_res['model']=='MLP'].drop('model',axis=1)
            
            GNN_res.to_csv(os.path.join(GNN_path,'csv',GNN_name+'.csv'))
            MLP_res.to_csv(os.path.join(MLP_path,'csv',MLP_name+'.csv'))
            
        CV_results = pd.concat(cv_results, axis=0, ignore_index=True)
        GNN_ROC = eval_.ROC_func(CV_results['GNNpred'],CV_results['label'],max_pred=1,intervals=1000)
        MLP_ROC = eval_.ROC_func(CV_results['MLPpred'],CV_results['label'],max_pred=1,intervals=1000)
        np.save(os.path.join(split_path, 'MLP_ROC_'+MLP_name), MLP_ROC)
        np.save(os.path.join(split_path, 'GNN_ROC_'+GNN_name), GNN_ROC)

        fig = plt.figure(figsize=(8, 6))
        tu.plot_roc('MLP',MLP_name,MLP_ROC)
        tu.plot_roc('GNN',GNN_name,GNN_ROC)
        plt.legend()
        plt.savefig(os.path.join(split_path, 'ShapeModels_ROC_'+MLP_name + '+' + GNN_name + '.png'))
        plt.show()
        plt.close()
        

def train_cv_shape_ensemble(home = '/media/mcgoug01/nvme/SecondYear/Data/',dataname='merged_training_set',
                         splits:list=[0],params:dict=None,folds=5):
    # Suppress warnings
    warnings.filterwarnings("ignore") #makes dgl stop complaining!

    # Initialization
    dev = tu.initialize_device()
    if params==None:params = tu.init_shape_params()
    else:tu.check_params(params,tu.init_shape_params())

    print(params)
    save_dir = tu.init_training_home(home, dataname)
    shapedataset, test_shapedataset = tu.get_shape_data(home, dataname, params['combined_threshold'], params['combined_threshold'],ensemble=True)
    cases, is_ncct = tu.get_cases(shapedataset)

    # More Initialization
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in splits:
        cv_results = []
        split_path = os.path.join(save_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)
            
        split_fp = os.path.join(split_path,'split.npy')
        five_fold_strat = kfold_strat(n_splits=folds, shuffle=True)

        if os.path.exists(split_fp):
            fold_split = np.load(split_fp,allow_pickle=True)
        else:  
            fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))],dtype=object)
            np.save(os.path.join(split_path,split_fp),fold_split)


        for fold,train_index, test_index in fold_split:
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            ensemble_path = os.path.join(fold_path,'shape_ensemble')
            MLP_path = os.path.join(fold_path,'MLP')
            GNN_path = os.path.join(fold_path,'GNN')
            dl,test_dl = tu.generate_dataloaders(shapedataset,test_shapedataset,cases[train_index],params['object_batchsize'],tu.shape_collate)

            if not os.path.exists(ensemble_path): os.mkdir(ensemble_path)
                
            MLP_name = '{}_{}_{}_{}'.format(params['s1_objepochs'],params['mlp_thresh'],params['mlp_lr'],params['object_batchsize'])
            MLP_mp = os.path.join(MLP_path,'model',MLP_name)
            MLP = torch.load(MLP_mp,map_location=dev)
            
            GNN_name = '{}_{}_{}_{}_{}_{}_{}'.format(params['s1_objepochs'],params['graph_thresh'],params['gnn_lr'],params['gnn_layers'],params['gnn_hiddendim'],params['gnn_neighbours'],params['object_batchsize'])
            GNN_mp = os.path.join(GNN_path,'model',GNN_name)
            GNN = torch.load(GNN_mp,map_location=dev)
            MLP.train(),GNN.train()

            ShapeEnsemble = model_generator.return_shapeensemble(MLP,GNN,n1=params['ensemble_n1'],n2=params['ensemble_n2'],num_labels=2,dev=dev)
            ShapeEnsemble.train()
            
            name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(params['ensemble_n1'],params['ensemble_n2'],params['shape_freeze_epochs'],params['shape_unfreeze_epochs'],params['graph_thresh'],params['mlp_thresh'],params['s1_objepochs'],params['shape_freeze_lr'],params['shape_unfreeze_lr'])
            if params['shape_freeze_epochs']>0:
                print('Frozen model training')
                SEopt = torch.optim.Adam(list(ShapeEnsemble.process1.parameters())+
                                              list(ShapeEnsemble.final.parameters())+
                                              list(ShapeEnsemble.MLP.layer3.parameters())+
                                              list(ShapeEnsemble.MLP.skip2.parameters())+
                                              list(ShapeEnsemble.GNN.classify2.parameters()),lr=params['shape_freeze_lr'])
    
                ShapeEnsemble = tu.train_shape_ensemble(dl,dev,params['shape_freeze_epochs'],loss_fnc,SEopt,ShapeEnsemble)
            if params['shape_unfreeze_epochs']>0:
                print('Unfrozen model training')
                SEopt = torch.optim.Adam(ShapeEnsemble.parameters(),lr=params['shape_unfreeze_lr'])
                ShapeEnsemble = tu.train_shape_ensemble(dl,dev,params['shape_unfreeze_epochs'],loss_fnc,SEopt,ShapeEnsemble)

            ShapeEnsemble.eval()

            if not os.path.exists(os.path.join(ensemble_path,'model')):
                os.mkdir(os.path.join(ensemble_path,'model'))
                os.mkdir(os.path.join(ensemble_path,'csv'))
            torch.save(ShapeEnsemble,os.path.join(ensemble_path,'model',name))
            ensemble_res,test_df = eval_.eval_shape_ensemble(ShapeEnsemble,test_dl,dev=dev)
            cv_results.append(test_df)

            ensemble_res.to_csv(os.path.join(ensemble_path,'csv',name+'.csv'))
            
        CV_results = pd.concat(cv_results, axis=0, ignore_index=True)
        ensemble_ROC = eval_.ROC_func(CV_results['pred'],CV_results['label'],max_pred=1,intervals=1000)
        np.save(os.path.join(split_path, 'Ensemble_ROC_'+name), ensemble_ROC)

        fig = plt.figure(figsize=(8, 6))
        tu.plot_roc('Shape Ensemble',name,ensemble_ROC)
        plt.legend()
        plt.savefig(os.path.join(split_path,'Ensemble_ROC_' +name+ '.png'))
        plt.show()
        plt.close()
        
if __name__ == '__main__':
    dataset = 'merged_training_set'
    # train_cv_individual_models(dataname=dataset)
    train_cv_shape_ensemble(dataname=dataset)
