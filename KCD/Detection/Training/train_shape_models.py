import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold as kfold_strat
import dgl
import KCD.Detection.Dataloaders.object_dataloader as dl_shape
import KCD.Detection.Evaluation.eval_scripts as eval_
from KCD.Detection.ModelGenerator import model_generator


def initialize_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def shape_collate(samples, dev=initialize_device()):
    features, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return torch.stack(features).to(dev), batched_graph.to(dev), torch.stack(labels).squeeze(dim=0).to(dev)


def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)


def init_params():
    params = {
        'tw_size': 'large',
        'sv_size': 'small',
        'gnn_layers': 5,
        'gnn_hiddendim': 25,
        'gnn_neighbours': 2,
        'object_batchsize': 8,
        'graph_thresh': 500,
        'mlp_thresh': 20000,
        'combined_threshold':500,
        'mlp_lr': 0.01,
        'gnn_lr': 0.001,
        's1_objepochs': 100,
        'shape_unfreeze_epochs': 2,
        'shape_freeze_epochs': 25,
        'shape_freeze_lr':1e-3,
        'shape_unfreeze_lr': 1e-3,
        'combined_threshold':500,
        'ensemble_n1':128,
        'ensemble_n2':32
    }
    return params


def check_params(params):
    inputset = set(params.keys())
    checkset = set(init_params.keys())
    assert(checkset.issubset(inputset))


def init_paths(home, dataname):
    save_dir = os.path.join(home, dataname)
    create_directory(home),create_directory(save_dir)
    return save_dir


def get_data(obj_path, dataname, graph_thresh, mlp_thresh,ensemble=False):
    shapedataset = dl_shape.ObjectData_labelled(obj_path, data_name=dataname, graph_thresh=graph_thresh, mlp_thresh=mlp_thresh,ensemble=ensemble)
    test_shapedataset = dl_shape.ObjectData_labelled(obj_path, data_name=dataname, graph_thresh=0, mlp_thresh=0,ensemble=ensemble)
    return shapedataset, test_shapedataset


def get_cases(shapedataset):
    cases = np.unique(shapedataset.cases.values)
    is_ncct = np.array([0 if case.startswith('case') else 1 for case in cases])
    return cases, is_ncct


def train_model(dl, dev, epochs, loss_fnc, opt, model):
    model.train()
    for i in range(epochs):
        print("\nEpoch {}".format(i))
        for features,graph,label in dl:
            pred = model(features.to(dev),graph.to(dev))
            if label.numel() != 1:label = label.squeeze()
            loss = loss_fnc(pred, label.to(dev))
            loss.backward()
            opt.step()
            opt.zero_grad()
            
    return model


def train_shape_models(dl, dev, epochs, loss_fnc, MLPopt, GNNopt, MLP, GNN):
    MLP.train(), GNN.train()
    for i in range(epochs):
        print("\nEpoch {}".format(i))
        for features, graph, lb in dl:
            feat_lb, graph_lb = lb.T
            MLPpred = MLP(features.to(dev))
            GNNpred = GNN(graph.to(dev))
            for pred, opt, label in zip([MLPpred, GNNpred], [MLPopt, GNNopt], [feat_lb, graph_lb]):
                if label.numel() == 1: label = label.unsqueeze(0)
                loss = loss_fnc(pred, label.to(dev))
                loss.backward()
                opt.step()
                opt.zero_grad()
                
    MLP.eval(),GNN.eval()
                
    return MLP,GNN


def generate_dataloaders(dataset,test_dataset,train_index,cases,batch_size,collate_fn):
    dataset.apply_foldsplit(train_cases = cases[train_index])
    test_dataset.apply_foldsplit(train_cases = cases[train_index])
    dl = DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    test_dl = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    return dl, test_dl


def plot_roc(name,model_name,ROC):
    sens, spec = ROC[:, 0] * 100, ROC[:, 1] * 100
    AUC = np.trapz(sens, spec) / 1e4
    plt.plot(100 - spec, sens, linewidth=2, label=name + ' AUC {:.3f}'.format(AUC))
    plt.ylabel('Sensivity / %')
    plt.xlabel('100 - Specificity / %')
    

def train_individual_models(home = '/Users/mcgoug01/Downloads/Data/',dataname='merged_training_set',
                            splits:list=[0],params:dict=None):
    # Suppress warnings
    warnings.filterwarnings("ignore") #makes dgl stop complaining!

    # Initialization
    dev = initialize_device()
    
    if params==None:params = init_params()
    else:check_params(params)
    
    save_dir = init_paths(os.path.join(home,'training_info'), dataname)
    shapedataset, test_shapedataset = get_data(home, dataname, params['graph_thresh'], params['mlp_thresh'],ensemble=True)
    cases, is_ncct = get_cases(shapedataset)

    # More Initialization
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in splits:
        cv_results = []
        split_path = os.path.join(save_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)
            
        split_fp = os.path.join(split_path,'split.npy')
        five_fold_strat = kfold_strat(n_splits=5, shuffle=True)

        if os.path.exists(split_fp):
            fold_split = np.load(split_fp,allow_pickle=True)
        else:  
            fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))])
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
    
            dl,test_dl = generate_dataloaders(shapedataset,test_shapedataset,train_index,cases,params['object_batchsize'],shape_collate)
            MLP,GNN = train_shape_models(dl,dev,params['s1_objepochs'],loss_fnc,MLPopt,GNNopt,MLP,GNN)
            
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
            
            GNN_res.to_csv(os.path.join(GNN_path,'csv',MLP_name+'.csv'))
            MLP_res.to_csv(os.path.join(MLP_path,'csv',GNN_name+'.csv'))
            
        CV_results = pd.concat(cv_results, axis=0, ignore_index=True)
        GNN_ROC = eval_.ROC_func(CV_results['GNNpred'],CV_results['label'],max_pred=1,intervals=1000)
        MLP_ROC = eval_.ROC_func(CV_results['MLPpred'],CV_results['label'],max_pred=1,intervals=1000)
        np.save(os.path.join(split_path, 'MLP_ROC_'+MLP_name), MLP_ROC)
        np.save(os.path.join(split_path, 'GNN_ROC_'+GNN_name), GNN_ROC)

        fig = plt.figure(figsize=(8, 6))
        plot_roc('MLP',MLP_name,MLP_ROC)
        plot_roc('GNN',GNN_name,GNN_ROC)
        plt.legend()
        plt.savefig(os.path.join(split_path, 'ShapeModels_ROC_'+MLP_name + '+' + GNN_name + '.png'))
        plt.show()
        plt.close()
        
        return MLP,GNN


def train_shape_ensemble(home = '/Users/mcgoug01/Downloads/Data/',dataname='merged_training_set',
                         splits:list=[0],params:dict=None):
    # Suppress warnings
    warnings.filterwarnings("ignore") #makes dgl stop complaining!

    # Initialization
    dev = initialize_device()
    if params==None:params = init_params()
    else:check_params(params)
    save_dir = init_paths(os.path.join(home,'training_info'), dataname)
    shapedataset, test_shapedataset = get_data(home, dataname, params['combined_threshold'], params['combined_threshold'],ensemble=True)
    cases, is_ncct = get_cases(shapedataset)

    # More Initialization
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in splits:
        cv_results = []
        split_path = os.path.join(save_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)
            
        split_fp = os.path.join(split_path,'split.npy')
        five_fold_strat = kfold_strat(n_splits=5, shuffle=True)

        if os.path.exists(split_fp):
            fold_split = np.load(split_fp,allow_pickle=True)
        else:  
            fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))])
            np.save(os.path.join(split_path,split_fp),fold_split)


        for fold,train_index, test_index in fold_split:
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            ensemble_path = os.path.join(fold_path,'shape_ensemble')
            MLP_path = os.path.join(fold_path,'MLP')
            GNN_path = os.path.join(fold_path,'GNN')
            dl,test_dl = generate_dataloaders(shapedataset,test_shapedataset,train_index,cases,params['object_batchsize'],shape_collate)

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
            
            name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(params['ensemble_n1'],params['ensemble_n2'],fold,params['shape_freeze_epochs'],params['shape_unfreeze_epochs'],params['graph_thresh'],params['mlp_thresh'],params['s1_objepochs'],params['shape_freeze_lr'],params['shape_unfreeze_lr'])
            if params['shape_freeze_epochs']>0:
                print('Frozen model training')
                SEopt = torch.optim.Adam(list(ShapeEnsemble.process1.parameters())+
                                              list(ShapeEnsemble.final.parameters())+
                                              list(ShapeEnsemble.MLP.layer3.parameters())+
                                              list(ShapeEnsemble.MLP.skip2.parameters())+
                                              list(ShapeEnsemble.GNN.classify2.parameters()),lr=params['shape_freeze_lr'])
    
                ShapeEnsemble = train_model(dl,dev,params['shape_freeze_epochs'],loss_fnc,SEopt,ShapeEnsemble)
            if params['shape_unfreeze_epochs']>0:
                print('Unfrozen model training')
                SEopt = torch.optim.Adam(ShapeEnsemble.parameters(),lr=params['shape_unfreeze_lr'])
                ShapeEnsemble = train_model(dl,dev,params['shape_unfreeze_epochs'],loss_fnc,SEopt,ShapeEnsemble)

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
        plot_roc('Shape Ensemble',name,ensemble_ROC)
        plt.legend()
        plt.savefig(os.path.join(split_path,'Ensemble_ROC_' +name+ '.png'))
        plt.show()
        plt.close()
        
        return ShapeEnsemble

if __name__ == '__main__':
    # MLP,GNN = train_individual_models()
    train_shape_ensemble()
