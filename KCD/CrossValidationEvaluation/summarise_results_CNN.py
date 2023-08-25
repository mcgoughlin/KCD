#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:54:31 2023

@author: mcgoug01
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/CNNKwise_cv_results'
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/high_level_results/CNNKwise'
dfs = []

for size in [fold for fold in os.listdir(path) if not fold.startswith('.')]:
    sizepath = os.path.join(path,size,'csv')
    save_path = os.path.join(save_dir,size) 
    size_dfs = []
    for results in [file for file in os.listdir(sizepath) if file.endswith('.csv')]:
        tempdf = pd.read_csv(os.path.join(sizepath,results))
        epochs = int(results.split('-')[-1].split('.')[0])
        tempdf['Epochs'] = [epochs]*len(tempdf)
        if 'Treshold' in tempdf.columns:
            tempdf['Threshold'] = tempdf['Treshold']
            tempdf = tempdf.drop('Treshold',axis=1)
    
        size_dfs.append( tempdf)
    size_df = pd.concat(size_dfs, axis=0, ignore_index=True)
    vox_vals = size_df.Voxel.unique()
    thresh_vals = size_df.Threshold.unique()
    vox_vals.sort()
    thresh_vals.sort()
    for column in [col for col in size_df.columns if not (col in ['Threshold','Voxel','Epochs','Voting Size','Type','dataset_loc'])]:
        col_matrix = size_df.groupby(['Threshold','Voxel','Epochs','Voting Size','Type','dataset_loc']).mean()[column].values.reshape(6,-1)
        col_df = pd.DataFrame(data=col_matrix,index=thresh_vals,columns=vox_vals,index=False)
        fp = os.path.join(save_path,'csv',column+".csv")
        print(fp)
        col_df.to_csv(fp)
        
        
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    AUC_surface = size_df.groupby(['Threshold','Voxel']).mean().AUC.values.reshape(-1)
    surf = ax.plot_trisurf(thresh_vals.tolist()*5, vox_vals.tolist()*7, AUC_surface, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(0.65,0.85)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    plt.savefig(os.path.join(save_path,'png','AUC_surface'))
    plt.close()
# average_df = sum(dfs)/len(dfs)
# average_df.to_csv(os.path.join(save_dir,'csv','results.csv'))

# for column in average_df.columns:
#     if 'num' in column.lower(): continue
#     plt.scatter(average_df.index,average_df[column])
#     plt.ylabel(column)
#     plt.xlabel('Training Label Threshold / mm cubed')
#     plt.show()
#     plt.savefig(os.path.join(save_dir,'png','{}.png'.format(column)))
#     plt.close()