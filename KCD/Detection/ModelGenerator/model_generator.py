#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:17:07 2023

@author: mcgoug01
"""

from torchvision import models
import torch
import torch.nn as nn
from dgl.nn import ChebConv
from modifiedGAP import GlobalAttentionPoolingPMG
import torch.nn.functional as F
import copy

class MLP_classifier(nn.Module):
    def __init__(self,num_features, num_labels, enc1_size=256, enc1_layers=3,
                 enc2_size=32, enc2_layers = 3, final_layers = 3,dev='cpu'):
        super(MLP_classifier, self).__init__()


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
        layer1 = self.dropout(self.actv(self.layer1(x)))
        layer2 = self.dropout(self.skip1(x) + self.actv(self.layer2(layer1)))
        return self.tanh(self.skip2(x) + self.actv(self.layer3(layer2)))

class GNN_classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,n_classes,neighbours=10,layers_deep = 4,device='cpu'):
        super(GNN_classifier, self).__init__()
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
   
class ShapeEnsemble(nn.Module):
    def __init__(self, MLP:MLP_classifier,GNN:GNN_classifier,
                 n1=128,n2=16,num_labels=2,device='cpu',dropout=0.8):
        super(ShapeEnsemble, self).__init__()
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

def return_efficientnet(size='small',dev='cpu',in_channels=1,out_channels=3):
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
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(dev)
    else:
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(dev)
        
    axial_tile_model.classifier[-1]= nn.Linear(1280,out_channels,bias=True).to(dev) 
    
    return axial_tile_model.to(dev)

def return_swin(size='small',dev='cpu',in_channels=1,out_channels=3):
    if size == 'small':
        model_generator = models.swin_v2_t
        weights = models.Swin_V2_T_Weights.IMAGENET1K_V1
    elif size == 'medium':
        model_generator = models.swin_v2_s
        weights = models.Swin_V2_S_Weights.IMAGENET1K_V1
    elif size == 'large':
        model_generator = models.swin_v2_b
        weights = models.Swin_V2_B_Weights.IMAGENET1K_V1
    else:
        assert(1==2)
    axial_tile_model = model_generator(weights = weights).to(dev)

    if size == 'large':
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,128,kernel_size=(4, 4), stride=(4, 4)).to(dev)
        axial_tile_model.head= nn.Linear(1024,out_channels,bias=True).to(dev) 
    else:
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,96,kernel_size=(4, 4), stride=(4, 4)).to(dev)
        axial_tile_model.head= nn.Linear(768,out_channels,bias=True).to(dev) 
    
    return axial_tile_model.to(dev)

def return_vit(size='small',dev='cpu',in_channels=1,out_channels=3):
    if size == 'small':
        model_generator = models.vit_b_16
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    elif size == 'medium':
        model_generator = models.vit_b_32
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1
    elif size == 'large':
        model_generator = models.vit_l_32
        weights = models.ViT_L_32_Weights.IMAGENET1K_V1
    else:
        assert(1==2)
        
    axial_tile_model = model_generator(weights = weights).to(dev)

    if size == 'large':
        axial_tile_model.conv_proj = nn.Conv2d(in_channels,1024,kernel_size=(32, 32), stride=(32, 32)).to(dev)
        axial_tile_model.heads.head= nn.Linear(1024,out_channels,bias=True).to(dev) 
    elif size == 'medium':
        axial_tile_model.conv_proj = nn.Conv2d(in_channels,768,kernel_size=(32, 32), stride=(32, 32)).to(dev)
        axial_tile_model.heads.head= nn.Linear(768,out_channels,bias=True).to(dev) 
    else:
        axial_tile_model.conv_proj = nn.Conv2d(in_channels,768,kernel_size=(16,16), stride=(16,16)).to(dev)
        axial_tile_model.heads.head= nn.Linear(768,out_channels,bias=True).to(dev) 
    
    return axial_tile_model.to(dev)

def return_resnet(size='small',dev='cpu',in_channels=1,out_channels=3):
    if size == 'small':
        model_generator = models.resnet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    elif size == 'medium':
        model_generator = models.resnet101
        weights = models.ResNet101_Weights.IMAGENET1K_V1
    elif size == 'large':
        model_generator = models.resnet152
        weights = models.ResNet152_Weights.IMAGENET1K_V1
    else:
        assert(1==2)

    axial_tile_model = model_generator(weights = weights).to(dev)
    
    axial_tile_model.conv1 = nn.Conv2d(in_channels,64,kernel_size=(4, 4), stride=(4, 4)).to(dev)
    axial_tile_model.fc= nn.Linear(2048,out_channels,bias=True).to(dev) 
    
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

def return_convnext(size='small',dev='cpu',in_channels=1,out_channels=3):
    if size == 'small':
        model_generator = models.convnext_small
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
    elif size == 'medium':
        model_generator = models.convnext_base
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    elif size == 'large':
        model_generator = models.convnext_large
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    else:
        assert(1==2)
        
        
    axial_tile_model = model_generator(weights = weights).to(dev)
    
    if size == 'large':
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,192,kernel_size=(4, 4), stride=(4, 4)).to(dev)
        axial_tile_model.classifier[-1]= nn.Linear(1536,out_channels,bias=True).to(dev) 
    elif size =='medium':
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,128, kernel_size=(4, 4), stride=(4, 4)).to(dev)
        axial_tile_model.classifier[-1]= nn.Linear(1024,out_channels,bias=True).to(dev) 
    else:
        axial_tile_model.features[0][0] = nn.Conv2d(in_channels,96, kernel_size=(4, 4), stride=(4, 4)).to(dev)
        axial_tile_model.classifier[-1]= nn.Linear(768,out_channels,bias=True).to(dev) 


    return axial_tile_model.to(dev)
    
def replace2d_to3d(layer,dev='cpu'):
    if str(type(layer))=="<class \'torch.nn.modules.linear.Linear\'>":
        return layer
    elif str(type(layer))=="<class \'torch.nn.modules.batchnorm.BatchNorm2d\'>":
        feat,mom,eps,aff = layer.num_features,layer.momentum,layer.eps,layer.affine
        # print(feat,mom,eps)
        return nn.BatchNorm3d(feat,eps,mom,affine=aff,device=dev)
    elif str(type(layer))=="<class \'torch.nn.modules.conv.Conv2d\'>":
        inc,outc,stride,pad,kern,is_bias = layer.in_channels,layer.out_channels,layer.stride[0],layer.padding[0],layer.kernel_size[0],layer.bias==None
        return nn.Conv3d(inc,outc,stride=stride,padding=pad,kernel_size=kern,bias=is_bias,device=dev)
    else:
        print(layer)
        assert(1==2)



def return_efficientnet3D(size='small',dev='cpu',in_channels=1,out_channels=3):
    eff2d = return_efficientnet(size,dev,in_channels,out_channels)
    
    completed_layers = []
    for name, _ in eff2d.named_parameters():
        # print(name)
        split_list = name.split('.')
        
        if len(split_list)==4:
            layname,id1,id2,_ = split_list
            blockname,id3,id4=[None]*3
        elif len(split_list)==7:    
            layname,id1,id2,blockname,id3,id4,_ = split_list
        elif len(split_list)==3:
            layname,id1,_ = split_list
            id2,blockname,id3,id4=[None]*4
        else:
            assert(1==2)
            
        layer_metadata = (id1,id2,blockname,id3,id4)
        if layer_metadata in completed_layers:continue
        completed_layers.append(layer_metadata)
        
        if layname=='features':
            if blockname!=None:
                if id4.isnumeric():
                    layer = eff2d.features[int(id1)][int(id2)].block[int(id3)][int(id4)]
                    
                    new_layer = replace2d_to3d(layer,dev=dev)
                    eff2d.features[int(id1)][int(id2)].block[int(id3)][int(id4)] = new_layer
                elif id4=='fc1':
                    layer = eff2d.features[int(id1)][int(id2)].block[int(id3)].fc1
                    new_layer = replace2d_to3d(layer,dev=dev)
                    eff2d.features[int(id1)][int(id2)].block[int(id3)].fc1 = new_layer
                elif id4=='fc2':
                    layer= eff2d.features[int(id1)][int(id2)].block[int(id3)].fc2
                    new_layer = replace2d_to3d(layer,dev=dev)
                    eff2d.features[int(id1)][int(id2)].block[int(id3)].fc2 = new_layer
                else:
                    assert(1==2)
            else:
                layer = eff2d.features[int(id1)][int(id2)]
                new_layer = replace2d_to3d(layer,dev=dev)
                eff2d.features[int(id1)][int(id2)] = new_layer
        elif layname=='classifier':
            layer = eff2d.classifier[int(id1)]
            new_layer = replace2d_to3d(layer,dev=dev)
            eff2d.classifier[int(id1)] = new_layer
            
        completed_layers.append(str(type(new_layer)))
        
    completed_layers = list(set(completed_layers))
    eff2d.avgpool = nn.AdaptiveMaxPool3d(output_size=1)
    
    return eff2d.to(dev)

def resnext_layerreplacer(layer):
    newlayer = copy.copy(layer)
    for subidx in range(len(layer)):
        sublayer = layer[subidx]
        newlayer[subidx].conv1 = replace2d_to3d(sublayer.conv1)
        newlayer[subidx].conv2 = replace2d_to3d(sublayer.conv2)
        newlayer[subidx].conv3 = replace2d_to3d(sublayer.conv3)
        newlayer[subidx].bn1 = replace2d_to3d(sublayer.bn1)
        newlayer[subidx].bn2 = replace2d_to3d(sublayer.bn2)
        newlayer[subidx].bn3 = replace2d_to3d(sublayer.bn3)
        if sublayer.downsample!=None:
            newlayer[subidx].downsample[0] = replace2d_to3d(sublayer.downsample[0])
            newlayer[subidx].downsample[1] = replace2d_to3d(sublayer.downsample[1])
        
    return newlayer
        

def return_resnext3D(size='small',dev='cpu',in_channels=1,out_channels=3):
    rnxt2d = return_resnext(size,dev,in_channels,out_channels)
    rnxt3d = copy.deepcopy(rnxt2d)
    completed_layers = []
    for name, _ in rnxt2d.named_parameters():
        # print(name)
        split_list = name.split('.')
        if len(split_list)==2:
            layname,_ = split_list
            blockname,id1,id2=[None]*3
        elif len(split_list)==4:    
            blockname,id1,layname,_ = split_list
            id2=None
        elif len(split_list)==5:
            blockname,id1,layname,id2,_ = split_list
        else:
            assert(1==2)
            
        layer_metadata = (blockname)
        if layer_metadata in completed_layers:continue
        completed_layers.append(layer_metadata)
        
        
        if blockname==None:
            layer1 = rnxt2d.conv1
            layer2 = rnxt2d.bn1
            layer3 = rnxt3d.fc
            rnxt3d.conv1 = replace2d_to3d(layer1,dev=dev)
            rnxt3d.bn1=replace2d_to3d(layer2,dev=dev)
            rnxt3d.fc=replace2d_to3d(layer3,dev=dev)
        elif blockname=='layer1':
            layer = rnxt2d.layer1
            rnxt3d.layer1 = resnext_layerreplacer(layer)
        elif blockname=='layer2':
            layer = rnxt2d.layer2
            rnxt3d.layer2 = resnext_layerreplacer(layer)
        elif blockname=='layer3':
            layer = rnxt2d.layer3
            rnxt3d.layer3 = resnext_layerreplacer(layer)
        elif blockname=='layer4':
            layer = rnxt2d.layer4
            rnxt3d.layer4 = resnext_layerreplacer(layer)            
        else:
            assert(1==2)
        
    completed_layers = list(set(completed_layers))
    rnxt3d.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    rnxt3d.avgpool = nn.AdaptiveMaxPool3d(output_size=1)
    
    return rnxt3d.to(dev)


def return_MLP(num_features=28, num_labels=2, enc1_size=128, enc1_layers=1,
             enc2_size=32, enc2_layers = 1, final_layers = 1,dev='cpu'):
    return MLP_classifier(num_features, num_labels, enc1_size, enc1_layers,
                 enc2_size, enc2_layers, final_layers,dev).to(dev)

def return_GNN(num_features=4,hidden_dim=50,num_labels=2,layers_deep=8,neighbours=8,dev='cpu'):
    return GNN_classifier(num_features,hidden_dim,num_labels,layers_deep,neighbours,dev).to(dev)


def return_shapeensemble(MLP,GNN,n1=128,n2=16,num_labels=2,dev='cpu'):
    return ShapeEnsemble(MLP,GNN,n1=n1,n2=n2,num_labels=num_labels,device=dev).to(dev)



    
