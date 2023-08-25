#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:12:29 2023

@author: mcgoug01
"""
import os 
import pandas as pd
import numpy as np
pred_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/inference_unseen/twcnn3d/final.csv'
label_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/all_add/features_stage1_4mm.csv'

preddf =pd.read_csv(pred_dir)
labeldf = pd.read_csv(label_dir)
labeldf['case']=labeldf['case'].str[:-7].str.upper()

preddf['label']=[0]*len(preddf)

vals = []
for case in np.unique(preddf.case):
    for position in np.unique(preddf[preddf.case==case].position):
        print(case,position)
        caselabel = labeldf[(labeldf.case==case)&(labeldf.position==position)]

        print(caselabel.cancer_0_vol.values)
        if len(caselabel)==0:
            vals.append({'case':case,'position':position,'label':'INVALID'})
            print('INVALID')
        elif caselabel.cancer_0_vol.values[0]>0:
            vals.append({'case':case,'position':position,'label':1,'size':caselabel.cancer_0_vol.values[0]})
        else:
            vals.append({'case':case,'position':position,'label':0,'size':caselabel.cancer_0_vol.values[0]})
           
valdf = pd.DataFrame(vals)
# preddf = pd.merge(preddf,valdf,on=['case','position'])
preddf['label'] = valdf.label
preddf['size'] = valdf['size']
# preddf.label=pd.(vals
preddf = preddf[preddf.label!='INVALID']
#case 137 and 001 have the R-L the wrong way around - correct in the labels manually.

preddf.to_csv('/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/inference_unseen/twcnn3d/compare.csv')

