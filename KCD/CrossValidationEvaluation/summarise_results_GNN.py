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
path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/GNN_cv_results'
save_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/high_level_results/GNN'
dfs = []

sizepath = os.path.join(path,'csv_4mm_final')
size_dfs = []
for results in [file for file in os.listdir(sizepath) if file.endswith('.csv')]:
    tempdf = pd.read_csv(os.path.join(sizepath,results))
    size_dfs.append( tempdf)
size_df = pd.concat(size_dfs, axis=0, ignore_index=True)
for column in [col for col in size_df.columns if not (col in ['Threshold','Epochs','Chebyshev neighbours','Hidden Dimensionality','Size'])]:
    col_matrix = size_df.groupby(['Threshold','Epochs','Chebyshev neighbours','Hidden Dimensionality','Size']).mean()[column]
    if (column == 'Validation AUC') or ('Sens' in column):
        col_matrix=pd.DataFrame(col_matrix)
        col_matrix['std']= size_df.groupby(['Threshold','Epochs','Chebyshev neighbours','Hidden Dimensionality','Size']).std()[column]
    fp = os.path.join(save_path,'csv_4mm',column+".csv")
    col_matrix.to_csv(fp)
        

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