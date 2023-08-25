#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:54:31 2023

@author: mcgoug01
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/ensemble/csv_V5'
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/high_level_results/ensemble'
indobj_dfs,twcnn_dfs,shape_dfs,all_dfs = [],[],[],[]
for results in [file for file in os.listdir(path) if (file.endswith('.csv'))]:
    tempdf = pd.read_csv(os.path.join(path,results))
    
    if results.startswith('IndObj'):
        indobj_dfs.append( tempdf)
    elif results.startswith('twCNN'):
        twcnn_dfs.append( tempdf)
    elif results.startswith('Shape'):
        shape_dfs.append( tempdf)
    elif results.startswith('Complete'):
        all_dfs.append( tempdf)
    else:
        assert(1==2)
        
        
for name,dfs in zip(['indobj','twcnn','shape','complete'],[indobj_dfs,twcnn_dfs,shape_dfs,all_dfs]):
    size_df = pd.concat(dfs, axis=0, ignore_index=True)
    size_df = size_df.drop(['fold','reading'],axis=1)
    size_df = size_df[size_df.dataset_loc=='test'] #filter to only show test results
    invariant_cols = [col for col in size_df.columns if size_df[col].values.min()==size_df[col].values.max()]
    size_df = size_df.drop(invariant_cols,axis=1)
    count_df = size_df.groupby([col for col in size_df.columns if not (('Highest' in col) or (col in ['AUC']))]).count().AUC
    size_df=size_df.groupby([col for col in size_df.columns if not (('Highest' in col) or (col in ['AUC']))]).mean()
    size_df['Count']=count_df.values
    for column in [col for col in size_df.columns if col!='Count']:
        col_matrix=size_df[[column,'Count']]
        fp = os.path.join(save_dir,'csv_V5','{}_{}results.csv'.format(name,column))
        col_matrix.to_csv(fp)
    
    # for column in average_df.columns:
    #     if 'num' in column.lower(): continue
    #     plt.scatter(average_df.index,average_df[column])
    #     plt.ylabel(column)
    #     plt.xlabel('Training Label Threshold / mm cubed')
    #     plt.show()
    #     plt.savefig(os.path.join(save_dir,'png','{}.png'.format(column)))
    #     plt.close()