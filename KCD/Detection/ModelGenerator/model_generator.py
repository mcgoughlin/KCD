#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:17:07 2023

@author: mcgoug01
"""

from torchvision import models
import torch
import torch.nn as nn
import numpy as np
from dgl.nn import ChebConv
from modifiedGAP import GlobalAttentionPoolingPMG
import torch.nn.functional as F

class MLP_classifier(nn.Module):
    def __init__(self,num_features, num_labels, enc1_size=256, enc1_layers=3,
                 enc2_size=32, enc2_layers = 3, final_layers = 3,dev='cpu'):
        super(MLP_classifier, self).__init__()
        # self.enc1_block = []
        # self.enc2_block = []
        # self.final_block = []
        # enc1_params = np.linspace(num_features, enc1_size, enc1_layers+1)
        # enc2_params = np.linspace(enc1_size, enc2_size, enc2_layers+1)
        # final_params = np.linspace(enc2_size, num_labels, final_layers+1)
        
        # for i in range(len(enc1_params)-1):
        #     self.enc1_block.append(nn.Linear(int(enc1_params[i]),int(enc1_params[i+1])).to(dev))
         
        # self.enc1_block = nn.ModuleList(self.enc1_block)
        # for i in range(len(enc1_params)-1):
        #     self.enc2_block.append(nn.Linear(int(enc2_params[i]),int(enc2_params[i+1])).to(dev))
        # self.enc2_block = nn.ModuleList(self.enc2_block)
        # for i in range(len(final_params)-1):
        #     self.final_block.append(nn.Linear(int(final_params[i]),int(final_params[i+1])).to(dev))
        # self.final_block = nn.ModuleList(self.final_block)

        self.layer1 = nn.Linear(num_features,enc1_size,bias= False).to(dev)
        
        self.skip1 = nn.Linear(num_features,enc2_size,bias= False).to(dev)
        self.layer2 = nn.Linear(enc1_size,enc2_size, bias= False).to(dev)
        
        self.skip2 = nn.Linear(num_features,num_labels,bias= False).to(dev)
        self.layer3 = nn.Linear(enc2_size,num_labels,bias=False).to(dev)
        self.actv = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.4)
        
    def forward(self,x):
        x = self.dropout(x)
        # x_comp = x
        # for layer in self.enc1_block:
        #     x_comp = self.actv(layer(x_comp))
            
        layer1 = self.dropout(self.actv(self.layer1(x)))
        # x_enc = skip1
        # for layer in self.enc2_block:
        #     x_enc = self.actv(layer(x_enc))
            
            
        layer2 = self.dropout(self.skip1(x) + self.actv(self.layer2(layer1)))
        return self.tanh(self.skip2(x) + self.actv(self.layer3(layer2)))

class Classifier_gen_original(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,n_classes,neighbours=10,layers_deep = 4,device='cpu'):
        super(Classifier_gen_original, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        self.conv1 = ChebConv(in_dim, hidden_dim_graph, neighbours).to(device)
        self.convs = nn.ModuleList([ChebConv(hidden_dim_graph, hidden_dim_graph, neighbours).to(device) for i in range(layers_deep-1)])
        pooling_gate_nn = nn.Linear(hidden_dim_graph, 1).to(device)
        self.layers_deep = layers_deep
        
        
        self.pooling = GlobalAttentionPoolingPMG(pooling_gate_nn).to(device)
        self.classify = nn.Linear(hidden_dim_graph,hidden_dim_graph).to(device)
        self.classify2 = nn.Linear(hidden_dim_graph, n_classes).to(device)
        self.hidden_dim_graph=hidden_dim_graph
        
        self.device=device
        
    def forward(self, g):
        g = g.to(torch.device(self.device))
        
        input_x = g.ndata['feat']
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, input_x))
        for i in range(self.layers_deep-1):
            h = F.relu(self.convs[i](g, h))
            
        # Calculate graph representation by averaging all the node representations.
        #hg = dgl.mean_nodes(g, 'h')
        [hg,g2] = self.pooling(g,h) 
        
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return a3
    
class ShapeEnsemble_V1(nn.Module):
    def __init__(self, MLP:MLP_classifier, CNN:models.efficientnet.EfficientNet,
                 GNN:Classifier_gen_original,n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(ShapeEnsemble_V1, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = MLP.to(device)
        self.CNN = CNN.to(device)
        self.GNN = GNN.to(device)
        
        self.n1=n1
        self.n2=n2
        
        # ensure all classifiers each output a vector of common dimensionality, dictated by n
        self.GNN.classify2 = nn.Linear(self.GNN.hidden_dim_graph,n1).to(device)
        self.CNN.classifier[1] = nn.Linear(self.CNN.classifier[1].in_features,n1).to(device)
        self.MLP.final_block[-1] = nn.Linear(self.MLP.final_block[-1].in_features,n1).to(device)
        self.MLP.skip3 = nn.Linear(self.MLP.skip3.in_features,n1).to(device)
        
        self.skip = nn.Linear(n1,n2).to(device)
        self.process1 = nn.Linear(n1,n2).to(device)
        self.process2 = nn.Linear(n2,n2).to(device)
        self.final = nn.Linear(n2,num_labels).to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        
    def forward(self, features,svim,graph):
        graph_enc = self.GNN(graph)
        cnn_enc = self.CNN(svim)
        mlp_enc = self.MLP(features)
        
        common_128 = self.actv(self.dropout(graph_enc+cnn_enc+mlp_enc))
        common_16 = self.actv(self.dropout(self.process1(common_128)))
        penultimate = self.actv(self.dropout(self.process2(common_16)+self.skip(graph_enc+cnn_enc+mlp_enc)))
        
        return self.final(penultimate)
    
class ShapeEnsemble_V2(nn.Module):
    def __init__(self, MLP:MLP_classifier, CNN:models.efficientnet.EfficientNet,
                 GNN:Classifier_gen_original,n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(ShapeEnsemble_V2, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = MLP.to(device)
        self.CNN = CNN.to(device)
        self.GNN = GNN.to(device)
        
        self.n1=n1
        self.n2=n2
        
        # ensure all classifiers each output a vector of common dimensionality, dictated by n
        self.GNN.classify2 = nn.Linear(self.GNN.hidden_dim_graph,n1).to(device)
        self.CNN.classifier[1] = nn.Linear(self.CNN.classifier[1].in_features,n1).to(device)
        self.MLP.final_block[-1] = nn.Linear(self.MLP.final_block[-1].in_features,n1).to(device)
        self.MLP.skip3 = nn.Linear(self.MLP.skip3.in_features,n1).to(device)
        
        self.process1 = nn.Linear(n1,n2).to(device)
        self.final = nn.Linear(n2,num_labels).to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        
    def forward(self, features,svim,graph):
        graph_enc = self.GNN(graph)
        cnn_enc = self.CNN(svim)
        mlp_enc = self.MLP(features)
        
        common_16 = self.actv(self.dropout(self.process1(graph_enc+cnn_enc+mlp_enc)))
        
        return self.final(common_16)
    
class ShapeEnsemble_V3(nn.Module):
    def __init__(self, MLP:MLP_classifier, CNN:models.efficientnet.EfficientNet,
                 GNN:Classifier_gen_original,n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(ShapeEnsemble_V3, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = MLP.to(device)
        self.CNN = CNN.to(device)
        self.GNN = GNN.to(device)
        
        self.n1=n1
        self.n2=n2
        
        # ensure all classifiers each output a vector of common dimensionality, dictated by n
        self.GNN.classify2 = nn.Linear(self.GNN.hidden_dim_graph,n1).to(device)
        self.CNN.classifier[1] = nn.Linear(self.CNN.classifier[1].in_features,n1).to(device)
        self.MLP.final_block[-1] = nn.Linear(self.MLP.final_block[-1].in_features,n1).to(device)
        self.MLP.skip3 = nn.Linear(self.MLP.skip3.in_features,n1).to(device)
        
        self.process1 = nn.Linear(n1*3,n2).to(device)
        self.final = nn.Linear(n2,num_labels).to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        
    def forward(self, features,svim,graph):
        graph_enc = self.GNN(graph)
        cnn_enc = self.CNN(svim)
        mlp_enc = self.MLP(features)
        enc = torch.cat([graph_enc,cnn_enc,mlp_enc],dim=-1)
        common_16 = self.actv(self.dropout(self.process1(enc)))
        
        return self.final(common_16)
    
class CompleteEnsemble_V1(nn.Module):
    def __init__(self, ShapeEnsemble:ShapeEnsemble_V1, tilewise_CNN:models.efficientnet.EfficientNet,
                 n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(CompleteEnsemble_V1, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = ShapeEnsemble.MLP.to(device)
        self.svCNN = ShapeEnsemble.CNN.to(device)
        self.GNN = ShapeEnsemble.GNN.to(device)
        self.twCNN = tilewise_CNN.to(device)
        self.n1=n1
        self.n2=n2

        self.tw_conv1 = nn.Conv1d(3,3,5,stride=1,padding=2).to(device) #padding ensures dimentionality of input doesn't change
        self.tw_conv2 = nn.Conv1d(3,1,15,stride=1,padding=7).to(device)
        self.tw_process = nn.Linear(self.n2,self.n2).to(device)
        
        self.process1 = ShapeEnsemble.process1.to(device)
        self.process2 = ShapeEnsemble.process2.to(device)
        self.final = ShapeEnsemble.final.to(device)
        self.skip = ShapeEnsemble.skip.to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        
    def forward(self, features,svim,graph,tilestack):
        graph_enc = self.GNN(graph)
        cnn_enc = self.svCNN(svim)
        mlp_enc = self.MLP(features)
        # there is a problem here - how do we integrate the kw tile classifiers with the shape ensemble..
        
        tile_stacks = torch.stack([self.twCNN(tile) for tile in tilestack]).squeeze()  #extract B x N x 3 features - N is num of tiles in case
        tile_stacks = torch.swapaxes(tile_stacks,-2,-1)
        tw_conv1 = self.actv(self.dropout(self.tw_conv1(tile_stacks))) # Convolve over features, output should be N x 3 again
        vals,indices = torch.topk(self.tw_conv2(tw_conv1),k=self.n2,dim=-1)  #Convolve over features, N x 1 
        tw_enc = self.tw_process(vals) # should be a vector of 16 features
        
        common_128 = self.actv(self.dropout(graph_enc+cnn_enc+mlp_enc))
        common_16 = self.actv(self.dropout(self.process1(common_128)+tw_enc.squeeze()))
        penultimate = self.actv(self.dropout(self.process2(common_16)+self.skip(graph_enc+cnn_enc+mlp_enc)+tw_enc.squeeze()))
        
        return self.final(penultimate)
    
class CompleteEnsemble_V2(nn.Module):
    def __init__(self, ShapeEnsemble:ShapeEnsemble_V2, tilewise_CNN:models.efficientnet.EfficientNet,
                 n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(CompleteEnsemble_V2, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = ShapeEnsemble.MLP.to(device)
        self.svCNN = ShapeEnsemble.CNN.to(device)
        self.GNN = ShapeEnsemble.GNN.to(device)
        self.twCNN = tilewise_CNN.to(device)
        self.n1=n1
        self.n2=n2

        self.tw_conv = nn.Conv1d(3,1,15,stride=1,padding=7).to(device)
        
        self.process1 = ShapeEnsemble.process1.to(device)
        self.final = ShapeEnsemble.final.to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        
    def forward(self, features,svim,graph,tilestack):
        graph_enc = self.GNN(graph)
        cnn_enc = self.svCNN(svim)
        mlp_enc = self.MLP(features)
        # there is a problem here - how do we integrate the kw tile classifiers with the shape ensemble..
        
        tile_stacks = torch.stack([self.twCNN(tile) for tile in tilestack]).squeeze()  #extract B x N x 3 features - N is num of tiles in case
        tile_stacks = torch.swapaxes(tile_stacks,-2,-1)
        tw_conv1 = self.actv(self.dropout(self.tw_conv(tile_stacks))) # Convolve over features, output should be N x 3 again
        vals,indices = torch.topk(tw_conv1,k=self.n2,dim=-1)  
        common_16 = self.actv(self.dropout(self.process1(graph_enc+cnn_enc+mlp_enc)+vals))
        
        return self.final(common_16)
    
class CompleteEnsemble_V3(nn.Module):
    def __init__(self, ShapeEnsemble:ShapeEnsemble_V2, tilewise_CNN:models.efficientnet.EfficientNet,
                 n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(CompleteEnsemble_V3, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = ShapeEnsemble.MLP.to(device)
        self.svCNN = ShapeEnsemble.CNN.to(device)
        self.GNN = ShapeEnsemble.GNN.to(device)
        self.twCNN = tilewise_CNN.to(device)
        self.n1=n1
        self.n2=n2

        self.tw_conv = nn.Conv1d(3,1,15,stride=1,padding=7,bias=False).to(device)
        self.tw_process = nn.Linear(n2,n2,bias=False).to(device)
        
        self.process1 = ShapeEnsemble.process1.to(device)
        self.final = ShapeEnsemble.final.to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        
    def forward(self, features,svim,graph,tilestack):
        graph_enc = self.GNN(graph)
        cnn_enc = self.svCNN(svim)
        mlp_enc = self.MLP(features)
        # there is a problem here - how do we integrate the kw tile classifiers with the shape ensemble..
        
        tile_stacks = torch.stack([self.twCNN(tile) for tile in tilestack]).squeeze()  #extract B x N x 3 features - N is num of tiles in case
        tile_stacks = torch.swapaxes(tile_stacks,-2,-1)
        tw_conv1 = self.actv(self.dropout(self.tw_conv(tile_stacks))) # Convolve over features, output should be N x 3 again
        vals,indices = torch.topk(tw_conv1,k=self.n2,dim=-1)  
        tw_enc = self.actv(self.tw_process(vals))
        common_16 = self.actv(self.dropout(self.process1(graph_enc+cnn_enc+mlp_enc)+tw_enc))
        
        return self.final(common_16)
    
class CompleteEnsemble_V4(nn.Module):
    def __init__(self, ShapeEnsemble:ShapeEnsemble_V3, tilewise_CNN:models.efficientnet.EfficientNet,
                 n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(CompleteEnsemble_V4, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = ShapeEnsemble.MLP.to(device)
        self.svCNN = ShapeEnsemble.CNN.to(device)
        self.GNN = ShapeEnsemble.GNN.to(device)
        self.twCNN = tilewise_CNN.to(device)
        self.n1=n1
        self.n2=n2

        self.tw_conv = nn.Conv1d(3,1,15,stride=1,padding=7).to(device)
        self.tw_process = nn.Linear(n2,n2).to(device)
        self.tw_pool = nn.AdaptiveMaxPool1d(n2)
        
        self.process1 = ShapeEnsemble.process1.to(device)
        self.process2 = nn.Linear(2*n2,n2).to(device)
        self.final = nn.Linear(n2,num_labels).to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        
    def forward(self, features,svim,graph,tilestack):
        graph_enc = self.GNN(graph)
        cnn_enc = self.svCNN(svim)
        mlp_enc = self.MLP(features)
        shape_enc = torch.cat([graph_enc,cnn_enc,mlp_enc],dim=-1)
        shape_n1 = self.actv(self.dropout(self.process1(shape_enc)))

        # there is a problem here - how do we integrate the kw tile classifiers with the shape ensemble..
        with torch.no_grad(): #prevent gradient flow through twCNN - reduce VRAM burden and unnecessary trainin
            tile_stacks = torch.stack([self.twCNN(tile) for tile in tilestack]).squeeze()  #extract B x N x 3 features - N is num of tiles in case
            tile_stacks = torch.swapaxes(tile_stacks,-2,-1)
        tw_conv1 = self.tw_conv(tile_stacks) # Convolve over features, output should be N x 3 again
        top_k = self.tw_pool(tw_conv1) 
        pre_squeeze = self.actv(self.dropout(self.tw_process(top_k)))
        if len(pre_squeeze.shape)==3:
            tw_enc = torch.squeeze(pre_squeeze,-2)
        elif len(pre_squeeze.shape)==2:
            tw_enc = pre_squeeze
        else:
            print(pre_squeeze.shape)
            assert(1==2)
        common_n2 = self.process2(torch.cat([shape_n1,tw_enc],dim=-1))
        return self.final(self.actv(common_n2))
    
class CompleteEnsemble_V5(nn.Module):
    def __init__(self, ShapeEnsemble:ShapeEnsemble_V2, tilewise_CNN:models.efficientnet.EfficientNet,
                 n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(CompleteEnsemble_V5, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = ShapeEnsemble.MLP.to(device)
        self.svCNN = ShapeEnsemble.CNN.to(device)
        self.GNN = ShapeEnsemble.GNN.to(device)
        self.twCNN = tilewise_CNN.to(device)
        self.n1=n1
        self.n2=n2

        self.tw_conv = nn.Conv1d(1,1,15,stride=1,padding=7).to(device)
        self.tw_process = nn.Linear(n2,n2).to(device)
        
        self.process1 = ShapeEnsemble.process1.to(device)
        self.final = ShapeEnsemble.final.to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features,svim,graph,tilestack):
        graph_enc = self.GNN(graph)
        cnn_enc = self.svCNN(svim)
        mlp_enc = self.MLP(features)
        # there is a problem here - how do we integrate the kw tile classifiers with the shape ensemble..
        with torch.no_grad(): #prevent gradient flow through twCNN - reduce VRAM burden and unnecessary trainin
            tile_stacks = torch.stack([self.twCNN(tile) for tile in tilestack]).squeeze(dim=1)  #extract B x N x 3 features - N is num of tiles in case
            tile_stacks = self.softmax(torch.swapaxes(tile_stacks,-2,-1))
            
        #extract only the prediction values for cancer
        tile_stacks = tile_stacks[:,2:]
        tw_conv1 = self.actv(self.dropout(self.tw_conv(tile_stacks))) # Convolve over features, output should be N x 3 again
        vals,indices = torch.topk(tw_conv1,k=self.n2,dim=-1)  
        tw_enc = self.actv(self.tw_process(vals)).squeeze(dim=-2)
        common_16 = self.actv(self.dropout(self.process1(graph_enc+cnn_enc+mlp_enc)+tw_enc))
        return self.final(common_16)
    
    
class ShapeEnsemble_noCNN(nn.Module):
    def __init__(self, MLP:MLP_classifier,
                 GNN:Classifier_gen_original,n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(ShapeEnsemble_noCNN, self).__init__()
        # Idea here is - a single ChebConv layer is like a single CNN layer with 1 filter of 10x10 width - we want 
        # multiple filters in parallel!
        
        self.MLP = MLP.to(device)
        self.GNN = GNN.to(device)
        
        self.n1=n1
        self.n2=n2
        
        # ensure all classifiers each output a vector of common dimensionality, dictated by n
        self.GNN.classify2 = nn.Linear(self.GNN.hidden_dim_graph,n1).to(device)
        self.MLP.layer3 = nn.Linear(self.MLP.layer3.in_features,n1).to(device)
        self.MLP.skip2 = nn.Linear(self.MLP.skip2.in_features,n1).to(device)
        
        self.process1 = nn.Linear(n1,n2).to(device)
        self.final = nn.Linear(n2,num_labels).to(device)
        self.dropout = nn.Dropout(dropout)
        self.actv = nn.ReLU()
        
        self.device=device
        
    def forward(self, features,graph):
        graph_enc = self.GNN(graph)
        mlp_enc = self.MLP(features)
        
        common_16 = self.actv(self.dropout(self.process1(graph_enc+mlp_enc)))
        
        return self.final(common_16)


def return_efficientnet(size='small',dev='cpu',in_channels=6,out_channels=2):
    if size == 'small':
        model_generator = models.efficientnet_v2_s
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    elif size == 'medium':
        model_generator = models.efficientnet_v2_m
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    elif size == 'large':
        model_generator = models.efficientnet_v2_l
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
    else:
        assert(1==2)
    axial_tile_model = model_generator(weights = weights).to(dev)
    
    if size == 'large':
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,32,3,2,1,bias=False).to(dev)
    else:
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,24,3,2,1,bias=False).to(dev)
        
    axial_tile_model.classifier[-1]= nn.Linear(1280,out_channels,bias=True).to(dev) 
    
    return axial_tile_model.to(dev)

def return_resnext(size='small',dev='cpu',in_channels=1,out_channels=3):
    if size == 'small':
        model_generator = models.resnext50_32x4d
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
    elif size == 'medium':
        model_generator = models.resnext101_64x4d
        weights = models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
    elif size == 'large':
        model_generator = models.resnext101_32x8d
        weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
    else:
        assert(1==2)
        
    axial_tile_model = model_generator(weights = weights).to(dev)

    axial_tile_model.conv1 = nn.Conv2d(in_channels,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(dev)
    axial_tile_model.fc= nn.Linear(2048,out_channels,bias=True).to(dev) 
    
    return axial_tile_model.to(dev)


def return_MLP(num_features=28, num_labels=2, enc1_size=128, enc1_layers=1,
             enc2_size=32, enc2_layers = 1, final_layers = 1,dev='cpu'):
    return MLP_classifier(num_features, num_labels, enc1_size, enc1_layers,
                 enc2_size, enc2_layers, final_layers,dev).to(dev)

def return_graphnn(num_features=4,hidden_dim=50,num_labels=2,layers_deep=8,neighbours=8,dev='cpu'):
    return Classifier_gen_original(num_features,hidden_dim,num_labels,layers_deep,neighbours,dev).to(dev)

def return_shapeensemble(CNN,MLP,GNN,n1=128,n2=16,num_labels=2,dev='cpu'):
    return ShapeEnsemble_V2(MLP,CNN,GNN,n1=n1,n2=n2,num_labels=num_labels,device=dev).to(dev)

def return_shapeensemble_noCNN(MLP,GNN,n1=128,n2=16,num_labels=2,dev='cpu'):
    return ShapeEnsemble_noCNN(MLP,GNN,n1=n1,n2=n2,num_labels=num_labels,device=dev).to(dev)

def return_completeensemble(ShapeEnsemble:ShapeEnsemble_V2, tilewise_CNN:models.efficientnet.EfficientNet,
             n1=128,n2=16,num_labels=2,dev='cpu'):
    return CompleteEnsemble_V5(ShapeEnsemble,tilewise_CNN,n1=n1,n2=n2,num_labels=num_labels,device=dev).to(dev)



    
