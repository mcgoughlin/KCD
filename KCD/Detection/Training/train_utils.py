import KCD.Detection.Dataloaders.object_dataloader as dl_shape
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import dgl
import matplotlib.pyplot as plt


def initialize_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
        
def init_paths(home, dataname):
    save_dir = os.path.join(home, dataname)
    create_directory(home),create_directory(save_dir)
    return save_dir
        

def shape_collate(samples, dev=initialize_device()):
    features, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return torch.stack(features).to(dev), batched_graph.to(dev), torch.stack(labels).squeeze(dim=0).to(dev)


def init_shape_params():
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
    checkset = set(init_shape_params().keys())
    assert(checkset.issubset(inputset))
    

def get_cases(dataset):
    cases = np.unique(dataset.cases.values)
    is_ncct = np.array([0 if case.startswith('case') else 1 for case in cases])
    return cases, is_ncct

def get_shape_data(obj_path, dataname, graph_thresh, mlp_thresh,ensemble=False):
    shapedataset = dl_shape.ObjectData_labelled(obj_path, data_name=dataname, graph_thresh=graph_thresh, mlp_thresh=mlp_thresh,ensemble=ensemble)
    test_shapedataset = dl_shape.ObjectData_labelled(obj_path, data_name=dataname, graph_thresh=0, mlp_thresh=0,ensemble=ensemble)
    return shapedataset, test_shapedataset


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