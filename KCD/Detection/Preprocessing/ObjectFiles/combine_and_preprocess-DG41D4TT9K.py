#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:32:02 2023

@author: mcgoug01
"""

import os
import numpy as np
import pandas as pd
import json
import shutil

ncct_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/kits_ncct'
add_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/coreg_ncct'
sncct_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/kits21sncct'
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/combined_dataset_21'

numpy_folders = ['curvatures','edges','vertices','2,2,2mm/fingerprints','2,2,2mm/images','2,2,2mm/predictions']

df_ncct = pd.read_csv(os.path.join(ncct_path,'features_stage2.csv'),index_col=0)
df_add = pd.read_csv(os.path.join(add_path,'features_stage2.csv'),index_col=0)
df_sncct = pd.read_csv(os.path.join(sncct_path,'features_stage2.csv'),index_col=0).dropna(subset='volume')

right = df_sncct[~df_sncct.case.str.split('_').str[1].str[:-7].isin(df_ncct.case.str.split('-').str[1].str[:-7].values)]

combined_kits = pd.merge(df_ncct,right,how='outer')
combined_all = pd.merge(combined_kits,df_add,how='outer')

cyst_df = combined_all.loc[:, combined_all.columns.str.contains('cyst')].fillna(0)
cancer_df = combined_all.loc[:, combined_all.columns.str.contains('cancer')].fillna(0)

combined_all['largest_cyst'] = cyst_df.max(axis=1)
combined_all['largest_cancer'] = cancer_df.max(axis=1)

skip_cols = ['case','position','largest_cyst','largest_cancer']
preprocessing = []
for column in combined_all.columns:
    
    if '_vol' in column:
        combined_all = combined_all.drop(column,axis='columns')
        continue
    if column in skip_cols: continue
    entry = {'column':column}
    entry['mean'] =combined_all[column].mean()
    entry['std'] =combined_all[column].std()
    preprocessing.append(entry)
    combined_all[column] = (combined_all[column] - combined_all[column].mean()) / (combined_all[column].std())

preprocessing = pd.DataFrame(preprocessing)

np.save(os.path.join(save_dir,'preprocessing_params.npy'),preprocessing)
preprocessing.to_csv(os.path.join(save_dir,'preprocessing_params.csv'))
combined_all.to_csv(os.path.join(save_dir,'preprocessed_features.csv'))


# transferring all combined files to the file dataset directory

obj_fold = os.path.join(save_dir,'cleaned_objs')
if not os.path.exists(obj_fold):
    os.mkdir(obj_fold)
    
fold = os.path.join(save_dir,'2,2,2mm')
if not os.path.exists(fold):
    os.mkdir(fold)
    
for folder in numpy_folders:
    np_fold = os.path.join(save_dir,folder)
    if not os.path.exists(np_fold):
        os.mkdir(np_fold)

for i,row in combined.iterrows():
    case = row['case']
    side = row['position']
    
    if side =='centre':continue
    
    if case.startswith('case'): fp = sncct_path
    else: fp = ncct_path
    
    numpy_name = case.split('.nii.gz')[0] +'_'+side+'.npy'
    casenumpy_name = case.split('.nii.gz')[0]+'.npy'
    obj_name = case.split('.nii.gz')[0] +'_'+side+'.obj'
    
    objfile = os.path.join(fp,'cleaned_objs',obj_name)
    objsave = os.path.join(save_dir,'cleaned_objs',obj_name)
    shutil.copy(objfile,objsave)
    
    for folder in numpy_folders:
        if folder.startswith('2,2,2'): name = casenumpy_name
        else: name = numpy_name
        npfile = os.path.join(fp,folder,name)
        npsave = os.path.join(save_dir,folder,name)
        shutil.copy(npfile,npsave)