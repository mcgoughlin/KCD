#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:12:29 2023

@author: mcgoug01
"""
import os 
import pandas as pd
import numpy as np
pred_dir = '/Users/mcgoug01/Downloads/Data/inference/merged_training_set_split_0/test_set/ensemble_128_32_25_2_500_20000_100_0.001_0.001.csv'
label_dir = '/Users/mcgoug01/Downloads/Data/objects/test_set/features_labelled.csv'

preddf =pd.read_csv(pred_dir)
labeldf = pd.read_csv(label_dir)
labeldf['case']=labeldf['case'].str[:-7].str.upper()

#make colum for  max cancer size from labeldf - cancer cols are cancer_0_vol, cancer_1_vol, cancer_2_vol... up to cancer_9_vol
labeldf['max_cancer_vol'] = labeldf[['cancer_{}_vol'.format(i) for i in range(10)]].max(axis=1)
labeldf['max_cancer_vol'] = labeldf['max_cancer_vol']
labeldf = labeldf[['case','position','max_cancer_vol']]
preddf['label']=[0]*len(preddf)

vals = []
for case in np.unique(labeldf.case):
    for position in np.unique(labeldf[labeldf.case==case].position):
        caselabel = labeldf[(labeldf.case==case)&(labeldf.position==position)].max_cancer_vol.values[0]>0
        if caselabel:
            vals.append({'case':case,'position':position,'label':1,'size':labeldf[(labeldf.case==case)&(labeldf.position==position)].max_cancer_vol.values[0]})
        else:
            vals.append({'case':case,'position':position,'label':0,'size':labeldf[(labeldf.case==case)&(labeldf.position==position)].max_cancer_vol.values[0]})
           
valdf = pd.DataFrame(vals)
preddf['label'] = valdf.label
preddf['size'] = valdf['size']

preddf.to_csv('/Users/mcgoug01/Downloads/Data/inference/merged_training_set_split_0/test_set/compare_shape.csv')
# shape label for R22, R56, R187 the wrong way around for some reason. Potentially others too, just haven't checked
# and are not relevant for the shape model, which predicts 0 for all other potentially incorrect cases.
