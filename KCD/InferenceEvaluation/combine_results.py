#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:54:31 2023

@author: mcgoug01
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/inference_unseen/twcnn3d/fold_wise_results/'
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/inference_unseen/twcnn3d/'
indobj_dfs,twcnn_dfs,shape_dfs,all_dfs = [],[],[],[]
for results in [file for file in os.listdir(path) if (file.endswith('.csv'))]:
    tempdf = pd.read_csv(os.path.join(path,results))
    
    if results.startswith('resnext3D') and results.endswith('10.csv'):
        twcnn_dfs.append( tempdf)
    else:
        pass
        
final_dfs =[]
for name,dfs in zip(['twcnn'],[twcnn_dfs]):
    model_df = pd.concat(dfs, axis=0, ignore_index=True)
    model_df = model_df.drop([column for column in model_df.columns if not (column  in ['case','position','prediction','pred-hard'])],axis=1)
    model_df = model_df.groupby(['case','position']).sum()
    model_df['output'] = (model_df>=2.5).astype(int)
    model_df.to_csv(os.path.join(save_dir,name)+'.csv')
    final_dfs.append(model_df)
    
final = pd.concat(final_dfs, axis=0)
final = final.groupby(['case','position']).sum()
final['tile_output']=(final['prediction']>2).astype(int)
final.to_csv(os.path.join(save_dir,'final')+'.csv')
    

    
    # for column in average_df.columns:
    #     if 'num' in column.lower(): continue
    #     plt.scatter(average_df.index,average_df[column])
    #     plt.ylabel(column)
    #     plt.xlabel('Training Label Threshold / mm cubed')
    #     plt.show()
    #     plt.savefig(os.path.join(save_dir,'png','{}.png'.format(column)))
    #     plt.close()