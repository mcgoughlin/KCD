import torch
import torch.nn as nn
import os
os.environ['OV_DATA_BASE'] ='/bask/projects/p/phwq4930-renal-canc/data/seg_data/'
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from random import random

from ne2ceCT.dataloader import CrossPhaseDataset
from ne2ceCT.loss_function import PyramidalLatentSimilarityLoss
from KCD.Segmentation.Inference import infer_network
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

def get_3d_featureoutput_unet(in_channels, out_channels, n_stages,filters=32, filters_max=1024):
    kernel_sizes = n_stages * [(3, 3, 3)]

    return infer_network.UNet_featureoutput(in_channels, out_channels, kernel_sizes, False, filters_max=filters_max, filters=filters, )

spacing = 4
epochs = 100
home_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/'
dataset_ne_path = os.path.join(home_path,'preprocessed', 'coltea','coltea_nat','coltea_nat_{}'.format(spacing))
dataset_ce_path = os.path.join(home_path,'preprocessed', 'coltea' 'coltea_art','coltea_art_{}'.format(spacing))

assert os.path.exists(dataset_ne_path) and os.path.exists(dataset_ce_path)
ne2ceCT_path = os.path.join(home_path, 'ne2ceCT','coltea_{}'.format(spacing))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_path = os.path.join(ne2ceCT_path, 'logs.txt')

if not os.path.exists(ne2ceCT_path):
    os.makedirs(ne2ceCT_path)


if os.path.exists(log_path):
    os.remove(log_path)


dataset = CrossPhaseDataset(os.path.join(dataset_ne_path,'images'), os.path.join(dataset_ce_path,'images'),
                            device=dev,
                            is_train=True,patches_per_case=20)

dataset.apply_foldsplit(split_ratio=0.9)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True,num_workers=5)


model_T = get_3d_featureoutput_unet(1, 2, 6,  filters=32, filters_max=1024).to(dev)
model_T.load_state_dict(torch.load(os.path.join(home_path,'trained_models','kits23_nooverlap','4mm_binary','for_jack_long','fold_1','network_weights')))
model_S = get_3d_featureoutput_unet(1, 2, 6,  filters=32, filters_max=1024).to(dev)
model_S.load_state_dict(torch.load(os.path.join(home_path,'trained_models','kits23_nooverlap','4mm_binary','for_jack_long','fold_1','network_weights')))

model_T.eval()
model_S.train()

optimizer = torch.optim.Adam(model_S.parameters(), lr=0.001)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
loss_func = PyramidalLatentSimilarityLoss()
losses = []
weight = 0.9
running_loss = 0.006
running_val_loss = 0.001
min_val_loss = 1
for epoch in range(epochs):
    dataloader.dataset.is_train = True
    model_S.train()
    for i, (ne, ce) in enumerate(dataloader):
        # forward
        ne = ne.to(dev)
        ce = ce.to(dev)
        with torch.no_grad():
            T = model_T(ce)
        S = model_S(ne)
        loss = loss_func(T, S)
        # backward
        loss.backward()
        # update weights
        optimizer.step()
        running_loss = weight * running_loss + (1 - weight) * loss.item()

        # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(dataloader), loss.item()))
        #print like above but with line replacement

        losses.append(running_loss)
        scheduler.step()
    train_string = f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss:.4E}, LR: {scheduler.get_last_lr()[0]:.4E}\n'
    with open(log_path, 'a') as f:
        f.write(string)

    dataloader.dataset.is_train = False
    with torch.no_grad():
        model_S.eval()
        for i, (ne, ce) in enumerate(dataloader):
            ne = ne.to(dev)
            ce = ce.to(dev)
            T = model_T(ce)
            S = model_S(ne)
            loss = loss_func(T, S)
            running_val_loss = weight * running_val_loss + (1 - weight) * loss.item()

    val_string = f'Val Loss: {running_val_loss:.4E}\n'
    with open(log_path, 'a') as f:
        f.write(val_string)
    dataloader.dataset.reset_patch_count()
    if running_val_loss < min_val_loss:
        min_val_loss = running_val_loss
        torch.save(model_S.state_dict(), os.path.join(ne2ceCT_path, 'network_weights'))
        print('Model saved')
        with open(log_path, 'a') as f:
            f.write('Model saved\n')

    with open(log_path, 'a') as f:
        f.write('\n')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(losses)
    ax.set_yscale('log')
    plt.savefig(os.path.join(ne2ceCT_path, 'losses.png'))
