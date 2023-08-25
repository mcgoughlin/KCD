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
dataset = 'combined_dataset_23andAdds'

path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/MLP_cv_results/csv_{}'.format(dataset)
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/high_level_results/MLP'
dfs = []
for results in [file for file in os.listdir(path) if (file.endswith('.csv'))]:
    tempdf = pd.read_csv(os.path.join(path,results))
    epochs = int(results.split('_')[0].split('epochs')[-1])
    if 'Treshold' in tempdf.columns:
        tempdf['Threshold'] = tempdf['Treshold']
        tempdf = tempdf.drop('Treshold',axis=1)
    tempdf['Epochs'] = [epochs]*len(tempdf)
    dfs.append( tempdf)
    
average_df = pd.concat(dfs, axis=0, ignore_index=True).groupby(['Threshold','Epochs']).mean().drop('Fold',axis=1)
average_df.to_csv(os.path.join(save_dir,'csv_{}'.format(dataset),'results_{}mm.csv'.format(spacing)))

for column in average_df.columns:
    if 'num' in column.lower(): continue
    plt.scatter(average_df.index,average_df[column])
    plt.ylabel(column)
    plt.xlabel('Training Label Threshold / mm cubed')
    plt.show()
    plt.savefig(os.path.join(save_dir,'png_{}'.format(dataset),'{}_{}mm.png'.format(column,spacing)))
    plt.close()