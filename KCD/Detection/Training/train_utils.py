import KCD.Detection.Dataloaders.object_dataloader as dl_shape
import KCD.Detection.Dataloaders.slice_dataloader as dl_slice

from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import dgl
import matplotlib.pyplot as plt


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
        'mlp_lr': 0.01,
        'gnn_lr': 0.001,
        's1_objepochs': 100,
        'shape_unfreeze_epochs': 2,
        'shape_freeze_epochs': 25,
        'shape_freeze_lr': 1e-3,
        'shape_unfreeze_lr': 1e-3,
        'combined_threshold': 500,
        'ensemble_n1': 128,
        'ensemble_n2': 32
    }
    return params


def init_slice2D_params():
    params = {"voxel_size": 1,
              "model_size": "large",
              "cancthresh_r_mm": 10,
              "kidthresh_r_mm": 20,
              "batch_size": 16,
              "dilated": 40,
              "lr": 5e-4,
              "epochs": 30,
              "depth_z": 1,
              "boundary_z": 1,
              'pred_window': 10}
    return params


def init_slice3D_params():
    params = {"voxel_size": 1,
              "model_size": "small",
              "cancthresh_r_mm": 10,
              "kidthresh_r_mm": 20,
              "batch_size": 16,
              "dilated": 40,
              "lr": 1e-3,
              "epochs": 30,
              "depth_z": 20,
              "boundary_z": 5,
              'pred_window': 1}
    return params

def init_slice3D_params_pretrain():
    params = {"voxel_size": 1,
              "model_size": "small",
              "cancthresh_r_mm": 10,
              "kidthresh_r_mm": 20,
              "batch_size": 16,
              "dilated": 40,
              "lr": 1e-3,
              "epochs": 5,
              "depth_z": 20,
              "boundary_z": 5,
              'pred_window': 1}
    return params

def init_slice3D_params_finetune():
    params = {"voxel_size": 1,
              "model_size": "small",
              "cancthresh_r_mm": 10,
              "kidthresh_r_mm": 20,
              "batch_size": 16,
              "dilated": 40,
              "lr": 5e-4,
              "epochs": 15,
              "depth_z": 20,
              "boundary_z": 5,
              'pred_window': 1}
    return params

def initialize_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
        
def init_training_home(home, dataname):
    training_home = os.path.join(home, 'training_info')
    save_dir = os.path.join(training_home, dataname)
    create_directory(home), create_directory(training_home), create_directory(save_dir)
    return save_dir
        

def shape_collate(samples, dev=initialize_device()):
    features, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return torch.stack(features).to(dev), batched_graph.to(dev), torch.stack(labels).squeeze(dim=0).to(dev)


def check_params(params,template_params):
    inputset = set(params.keys())
    checkset = set(template_params.keys())
    assert(checkset.issubset(inputset))
    

def get_cases(dataset):
    cases = np.unique(dataset.cases)
    is_ncct = np.array([0 if case.startswith('case') else 1 for case in cases])
    return cases, is_ncct


def get_shape_data(home, dataname, graph_thresh, mlp_thresh,ensemble=False,dev='cpu'):
    shapedataset = dl_shape.ObjectData_labelled(home, data_name=dataname, graph_thresh=graph_thresh, mlp_thresh=mlp_thresh,ensemble=ensemble,dev=dev)
    test_shapedataset = dl_shape.ObjectData_labelled(home, data_name=dataname, graph_thresh=0, mlp_thresh=0,ensemble=ensemble,dev=dev)
    return shapedataset, test_shapedataset


def get_slice_data(home, dataname, voxel_size, cancthresh,kidthresh,depth_z,
                   boundary_z,dilated,device='cpu'):
    slicedataset = dl_slice.SW_Data_labelled(home, dataname, voxel_size_mm=voxel_size, cancthresh=cancthresh,kidthresh=kidthresh,depth_z=depth_z,boundary_z=boundary_z,dilated=dilated,device=device)
    test_slicedataset = dl_slice.SW_Data_labelled(home, dataname, voxel_size_mm=voxel_size, cancthresh=0,kidthresh=0,depth_z=depth_z,boundary_z=boundary_z,dilated=dilated,device=device)
    return slicedataset, test_slicedataset


def train_shape_ensemble(dl, dev, epochs, loss_fnc, opt, model):
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


def train_model(dl, dev, epochs, loss_fnc, opt, model):
    model.train()
    for i in range(epochs):
        print("\nEpoch {}".format(i))
        for features,label in dl:
            pred = model(features.to(dev))
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


def generate_dataloaders(dataset,test_dataset,train_cases,batch_size,collate_fn=None):
    dataset.apply_foldsplit(train_cases = train_cases)
    test_dataset.apply_foldsplit(train_cases = train_cases)
    dl = DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    test_dl = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    return dl, test_dl


def plot_roc(name,model_name,ROC):
    sens, spec = ROC[:, 0] * 100, ROC[:, 1] * 100
    AUC = np.trapz(sens, spec) / 1e4
    plt.plot(100 - spec, sens, linewidth=2, label=name + ' AUC {:.3f}'.format(AUC))
    plt.ylabel('Sensitivity / %')
    plt.xlabel('100 - Specificity / %')