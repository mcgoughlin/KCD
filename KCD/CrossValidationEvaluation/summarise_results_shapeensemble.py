#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:54:31 2023

@author: mcgoug01
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/shape_ensemble'
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/high_level_results/shape_ensemble'
dfs = []
for fold in [file for file in os.listdir(path) if (file.startswith('fold'))]:
    fold_fp = os.path.join(path,fold,'ShapeEnsemble','csv')
    for results in [file for file in os.listdir(fold_fp) if (file.endswith('.csv'))]:
        _,_,n1,n2,fold,fe,ufe,gt,mt,se,flr,uflr= results.split('_')
        tempdf = pd.read_csv(os.path.join(fold_fp,results))
        tempdf['ufe'] = [ufe]*len(tempdf)
        tempdf['flr'] = [flr]*len(tempdf)
        
        dfs.append(tempdf)
        
        
size_df = pd.concat(dfs, axis=0, ignore_index=True)
size_df = size_df[size_df.dataset_loc=='test'] #filter to only show test results
invariant_cols = [col for col in size_df.columns if size_df[col].values.min()==size_df[col].values.max()]
size_df = size_df.drop(invariant_cols,axis=1)
size_df=size_df.groupby([col for col in size_df.columns if not (('Highest' in col) or ('Boundary' in col) or (col in ['AUC']))]).mean()
for column in [col for col in size_df.columns if not ('Boundary' in col)]:
    col_matrix=size_df[[column]]
    fp = os.path.join(save_dir,'csv','{}results.csv'.format(column))
    col_matrix.to_csv(fp)
    
    # for column in average_df.columns:
    #     if 'num' in column.lower(): continue
    #     plt.scatter(average_df.index,average_df[column])
    #     plt.ylabel(column)
    #     plt.xlabel('Training Label Threshold / mm cubed')
    #     plt.show()
    #     plt.savefig(os.path.join(save_dir,'png','{}.png'.format(column)))
    #     plt.close()