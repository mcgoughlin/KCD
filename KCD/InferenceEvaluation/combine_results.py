#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:54:31 2023

@author: mcgoug01
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

lab_path = '/Users/mcgoug01/Downloads/Data/inference/label.csv'
path_3d = '/Users/mcgoug01/Downloads/Data/inference/coreg_ncct_split_0/test_set/PatchModel_TESTMODEL_RESNEXT_kits23_nooverlap_PatchModel_small_5_5_0.0005.csv'
save_loc = '/Users/mcgoug01/Downloads/Data/inference/coreg_ncct_split_0/test_set/combined_results_resnext.csv'

lab_df = pd.read_csv(lab_path,index_col=False)
#convert lab_df position to 'right' if 1 and 'left' if 0
lab_df['position'] = lab_df['position'].apply(lambda x: 'right' if x==1 else 'left')
pred_3d = pd.read_csv(path_3d,index_col=False)
# drop columns that contain 'unnamed'
pred_3d = pred_3d.drop([column for column in pred_3d.columns if 'unnamed' in column.lower()],axis=1)
#combine on label's case and position
combined = pd.merge(lab_df,pred_3d,on=['case','position'],how='outer')
#fill with 0s prediction
combined['prediction'] = combined['prediction'].fillna(0)
#save
combined.to_csv(save_loc,index=False)