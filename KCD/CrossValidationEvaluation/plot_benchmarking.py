#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:25:37 2023

@author: mcgoug01
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import numpy as np

data_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/high_level_results/benchmarking/csv'
epochs = [1,3,4,5,10,15,20]
sizes = ['small','medium','large']
voting_sizes = [5,10,15]
columns = ['AUC','Highest Cancer Sens @ 98% Cancer Spec']

models = [model for model in os.listdir(data_dir) if not '.' in model]
colours = cm.winter_r(np.linspace(0, 1, len(sizes)))

for column in columns:

    for epoch in epochs:
        for voting_size in voting_sizes: 
            results = []
        
            for model in models:
                AUCfp = os.path.join(data_dir,model,'csv','{}.csv'.format(column))
                AUCdf = pd.read_csv(AUCfp)
                AUCdf = AUCdf[(AUCdf['Type']=='Top')]
                votingdf = AUCdf[AUCdf['Voting Size']==voting_size]
            
                epochdf = votingdf[votingdf['Epochs']==epoch]
                for colour,size in zip(colours,sizes):
                    try:
                        entry = {'model':model,
                                 'epoch':epoch,
                                 'voting_size':voting_size,
                                 'size':size[0].upper() + size[1:],
                                     column:epochdf[epochdf['size']==size][column].values[0],
                                 'colour':colour,
                                 'std':epochdf[epochdf['size']==size]['std'].values[0]
                                 }
                    except:
                        continue
                    results.append(entry)
                        
            resultsdf = pd.DataFrame(results)
            resultsdf = resultsdf.sort_values(['size','model'],ascending=True)
            fig = plt.figure(figsize=(10,6))
            # AUCplot = sns.scatterplot(data=resultsdf, x="model", y=column, hue="model",size='size',edgecolor='black',alpha=(epoch+10)/30,linewidth=0.7)
            AUCplot = sns.scatterplot(data=resultsdf, x="model", y=column, hue="model",size='size',edgecolor='black',linewidth=0.7,sizes=[200,100,50])
    
            h,l = AUCplot.get_legend_handles_labels()
            plt.xlabel('Image Classifier',fontsize=18)
            plt.ylabel('Case-wise {}'.format(column),fontsize=18)
            if column == 'AUC':
                AUCplot.legend(handles=h[-3:],labels=l[-3:],bbox_to_anchor=(0.71,-0.11),ncols=3)
                plt.ylim((0.5,1.05))
            else:
                AUCplot.legend(handles=h[-3:],labels=l[-3:],bbox_to_anchor=(1,1))
                plt.ylim((0,1.0))
            plt.show()
            plt.savefig(os.path.join(data_dir,'{}_ep{}_vs{}.png'.format(column,epoch,voting_size)),bbox_inches='tight')
            plt.close()
