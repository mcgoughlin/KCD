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
path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/benchmarking_3d/csv'
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/high_level_results/benchmarking_3d'
dfs = []

size_dfs = []
for results in [file for file in os.listdir(path) if file.endswith('.csv')]:
    try:
        tempdf = pd.read_csv(os.path.join(path,results))
    except:
        print(results)
        continue
    model,size,epochs,fold,reading = results.split('_')
    epochs = int(epochs.split('epochs')[-1])
    fold = int(fold[-1])
    reading = int(reading.split('.')[0][-1])

    tempdf['Epochs'] = [epochs]*len(tempdf)
    tempdf['fold'] = [fold]*len(tempdf)
    tempdf['reading'] = [reading]*len(tempdf)
    tempdf['model'] = [model]*len(tempdf)
    tempdf['size'] = [size]*len(tempdf)
        
    
    size_dfs.append( tempdf[tempdf.dataset_loc=='test'].drop(['dataset_loc'],axis=1))
size_df = pd.concat(size_dfs, axis=0, ignore_index=True)
for model_name in size_df.model.unique():
    final_df = size_df[size_df['model']==model_name].drop(['model','reading','fold'],axis=1)

    for column in [col for col in final_df.columns if not (col in ['Epochs','Voting Size','Type','model','size'])]:
        col_matrix = final_df.groupby(['Epochs','Voting Size','Type','size']).mean()[column]
        count_df = final_df.groupby(['Epochs','Voting Size','Type','size']).count().AUC.rename('count')
        col_matrix =pd.concat([col_matrix, count_df], axis=1)
        if (column == 'AUC') or ('Sens' in column):
            col_matrix['std']= final_df.groupby(['Epochs','Voting Size','Type','size']).std()[column]
        fp = os.path.join(save_dir,model_name,'csv',column+".csv")
        
        
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