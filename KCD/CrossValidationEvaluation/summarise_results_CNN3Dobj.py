#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:54:31 2023

@author: mcgoug01
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

spacing = 4
dataset = 'combined_dataset_23'
size = 'small'

path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/CNN_3D_Kwise_cv_results/{}/{}/csv_{}mm'.format(dataset,
                                                                                                                                                          size,spacing)
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/high_level_results/CNN_3Dobj'
dfs = []
for results in [file for file in os.listdir(path) if (file.endswith('.csv'))]:
    tempdf = pd.read_csv(os.path.join(path,results))
    if 'Treshold' in tempdf.columns:
        tempdf['Threshold'] = tempdf['Treshold']
        tempdf = tempdf.drop('Treshold',axis=1)

    dfs.append( tempdf.groupby(['Threshold','Epochs','dataset_loc']).mean())
    
    
    
average_df = pd.concat(dfs).groupby(['Threshold','Epochs','dataset_loc']).mean().reset_index()
average_df = average_df[average_df['dataset_loc']=='test']
average_df.to_csv(os.path.join(save_dir,'csv_{}_{}mm'.format(dataset,spacing),'results_{}.csv'.format(size)))

for column in average_df.columns:
    if 'num' in column.lower(): continue
    plt.scatter(average_df.index,average_df[column])
    plt.ylabel(column)
    plt.xlabel('Training Label Threshold / mm cubed')
    plt.show()
    plt.savefig(os.path.join(save_dir,'png_{}_{}mm'.format(dataset,spacing),'{}_{}.png'.format(size,column)))
    plt.close()