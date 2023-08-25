#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:03:25 2023

@author: mcgoug01
"""

import os 
import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

res = pd.read_csv('/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/inference_unseen/twcnn3d/compare_assessed.csv')
res2d = pd.read_csv('/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/inference_unseen/twcnn2d/compare_assessed_tilestackcorrected.csv')
res2d['2d_output'] = res2d['tile_output']
res2d = res2d [['2d_output','Shape_output','case','position','label']]

res = pd.merge(res,res2d,how='outer',on=['case','position','label'])
res['2dOR_output']=((res['2d_output']==1) | (res.Shape_output==1)).astype(int)
res['3dOR_output']=((res['tile_output']==1) | (res.Shape_output==1)).astype(int)
output_cols =['2d_output','tile_output','Shape_output','2dOR_output','3dOR_output']
res = res[res.label.notnull()]
for col in output_cols:
    output = res[col]
    tp = sum((output==res['label']) & (output==1))
    tn = sum((output==res['label']) & (output==0))
    fp = sum((output!=res['label']) & (output==1))
    fn = sum((output!=res['label']) & (output==0))
    
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    print(col,tp,tn,fp,fn,sens,spec)
    
# size distribution of cancer : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3271446/#:~:text=Tumor%20size%20is%20not%20an,the%20histological%20subtype%20of%20RCC

sizes = np.array([20,30,40,50,60,70,80])/2

F1 = np.array([5,8,5,3,1,0,2])/(24)
F2 = np.array([17,64,83,35,37,8,22])/(266)
F3 = np.array([7,20,43,30,26,19,40])/(185)
F4 = np.array([0,0,7,3,3,5,13])/(31)

fig = plt.figure(figsize=(10,6))
plt.plot(sizes,F1,label='Fuhrman Grade 1')
plt.plot(sizes,F2,label='Fuhrman Grade 2')
plt.plot(sizes,F3,label='Fuhrman Grade 3')
plt.plot(sizes,F4,label='Fuhrman Grade 4')
plt.ylim(0,1)
plt.xlim(10,40)
plt.legend()
plt.xlabel('Equivalent Radius / mm',fontsize=18)
plt.ylabel('Prevalance / %',fontsize=18)
    
smoothed = []
for col in  output_cols:
    res['size'] = res['size'].fillna(0)
    results = []
    rad_list =[5,10,15,20,25,30,35,1000]
    for i in range(len(rad_list)):
        rad = rad_list[i]
    
        if rad==1000: continue
        rad_ahead = rad_list[i+1]
        vol = 1.333*3.1416*(rad**3)
        vol_ahead = 1.333*3.1416*(rad_ahead**3)
        size_res = res[(res[col].notnull()) & (res['size']<vol_ahead)]
        
        tp = sum((size_res[col]==1) & (size_res['size']>=vol) )
        tn = sum((size_res[col]==0) & (size_res['size']<vol))
        fp = sum((size_res[col]==1) & (size_res['size']==0))
        fn = sum((size_res[col]==0) & (size_res['size']>=vol))
        sens = tp/(tp+fn+0.01)
        spec = tn/(tn+fp+0.01)
        results.append([rad,sens,spec])
        # assert(1==2)
        
    results = np.array(results)

    ysmoothed = gaussian_filter1d(results[:,1], sigma=1)
    smoothed.append(ysmoothed)
    
doc = np.array([0.28,0.28,0.895,0.895,0.895,0.895,0.895])
    
    
for name,probability in zip(output_cols,[*smoothed,doc]):
    print(name)
    print('Stage 1: {:.2f}%'.format(sum(probability*F1)*100))
    print('Stage 2: {:.2f}%'.format(sum(probability*F2)*100))
    print('Stage 3: {:.2f}%'.format(sum(probability*F3)*100))
    print('Stage 4: {:.2f}%'.format(sum(probability*F4)*100))
        
