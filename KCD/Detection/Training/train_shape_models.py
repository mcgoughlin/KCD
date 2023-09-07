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
        'mlp_lr': 0.01,
        'gnn_lr': 0.001,
        's1_objepochs': 20,
        'dataname': 'merged_training_set'
    }
    return params


def init_paths(home, dataname):
    save_dir = os.path.join(home, dataname)
    create_directory(home),create_directory(save_dir)
    return save_dir


def get_data(obj_path, dataname, graph_thresh, mlp_thresh):
    shapedataset = dl_shape.ObjectData_labelled(obj_path, data_name=dataname, graph_thresh=graph_thresh, mlp_thresh=mlp_thresh)
    test_shapedataset = dl_shape.ObjectData_labelled(obj_path, data_name=dataname, graph_thresh=0, mlp_thresh=0)
    return shapedataset, test_shapedataset


def get_cases(shapedataset):
    cases = np.unique(shapedataset.cases.values)
    is_ncct = np.array([0 if case.startswith('case') else 1 for case in cases])
    return cases, is_ncct


def train_model(dl, dev, epochs, loss_fnc, opt, model):
    model.train()
    for i in range(epochs):
        print("\nEpoch {}".format(i))
        for features,label in dl:
            pred = model(features.to(dev))
            if label.numel() == 1:
                label = label.unsqueeze(0)
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
                if label.numel() == 1:
                    label = label.unsqueeze(0)
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
    

def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Initialization
    dev = initialize_device()
    params = init_params()
    home = '/Users/mcgoug01/Downloads/Data/training_info'
    obj_path = '/Users/mcgoug01/Downloads/Data/'
    save_dir = init_paths(home, params['dataname'])
    shapedataset, test_shapedataset = get_data(obj_path, params['dataname'], params['graph_thresh'], params['mlp_thresh'])
    cases, is_ncct = get_cases(shapedataset)

    # More Initialization
    results = []
    softmax = nn.Softmax(dim=-1)
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in [0,1,2]:
        test_res,train_res = [], []
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
        plt.savefig(os.path.join(split_path, MLP_name + '+' + GNN_name + '.png'))
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
