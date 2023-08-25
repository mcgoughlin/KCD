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
res2d = res2d [['Shape_output','case','position','label','Contour (bill)','2d_output']]

res = pd.merge(res,res2d,how='outer',on=['case','position','label'])
res['2dOR_output']=((res['2d_output']==1) | (res.Shape_output==1)).astype(int)
res['3dOR_output']=((res.tile_output==1) | (res.Shape_output==1)).astype(int)
res['OR_all_output']=((res['3dOR_output']==1) | (res['2d_output']==1)).astype(int)
output_cols =['tile_output','Shape_output','2dOR_output','3dOR_output','2d_output','OR_all_output']

for col in output_cols:
    output = res[col]
    tp = sum((output==res['label']) & (output>0))
    tn = sum((output==res['label']) & (output<1))
    fp = sum((output!=res['label']) & (output>0))
    fn = sum((output!=res['label']) & (output<1))
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    print(col,len(output),(tn+tp+fn+fp),sens,spec)
    
fig = plt.figure(figsize=(10,6))
for col in ['tile_output','Shape_output','2dOR_output','3dOR_output','2d_output']:
    res = res.fillna(0)
    results = []
    rad_list =[5,10,15,20,25,30,35,1000]
    for i in range(len(rad_list)):
        rad = rad_list[i]
    
        if rad==1000: continue
        rad_ahead = rad_list[i+1]
        vol = 1.333*3.1416*(rad**3)
        vol_ahead = 1.333*3.1416*(rad_ahead**3)
        size_res = res.fillna(0)
        size_res = res[res['size']<vol_ahead]
        
        tp = sum((size_res[col]==1) & (size_res['size']>=vol) )
        tn = sum((size_res[col]==0) & (size_res['size']<vol))
        fp = sum((size_res[col]==1) & (size_res['size']==0))
        fn = sum((size_res[col]==0) & (size_res['size']>=vol))
        sens = tp/(tp+fn+0.01)
        spec = tn/(tn+fp+0.01)
        results.append([rad+5,sens,spec])
        # assert(1==2)
        
    results = np.array(results)

    ysmoothed = gaussian_filter1d(results[:,1], sigma=1)

    if col == 'tile_output':
        # plt.scatter(results[:,0],results[:,1]*100,700,'indianred',marker='.',label='3D Patchwise CNN')
        # plt.plot(results[:,0],results[:,2]*100,'-r',linewidth=1) 
        plt.plot(results[:,0]*2,ysmoothed*100,'-',c='indianred',linewidth=3)
        plt.show()
    elif col =='2dOR_output':
        # pass
        # plt.scatter(results[:,0],results[:,1]*100,400,'mediumorchid',marker='.',label='2D OR Combination')
        # plt.plot(results[:,0],results[:,2]*100,'-k',linewidth=1)        # a =results.tolist()
        plt.plot(results[:,0]*2,ysmoothed*100,'-.',c='mediumorchid',linewidth=3)
        # del(a[3])
        # results = np.array(a)
        # plt.plot(results[:,0],results[:,1]*100,'-.k',linewidth=1.5,label='Interpolated Sensitivity OR')
    elif col =='3dOR_output':
        pass
        # plt.scatter(results[:,0],results[:,1]*100,250,'rebeccapurple',marker='.',label='3D OR Combination')
        # plt.plot(results[:,0],results[:,2]*100,'-k',linewidth=1)        # a =results.tolist()
        # plt.plot(results[:,0],ysmoothed*100,'--',c='rebeccapurple',linewidth=1.5)
    elif col =='2d_output':
        # plt.scatter(results[:,0],results[:,1]*100,125,'lightsalmon',marker='.',label='2D Tilewise Combination')
        # plt.plot(results[:,0],results[:,2]*100,'-k',linewidth=1)        # a =results.tolist()
        plt.plot(results[:,0]*2,ysmoothed*100,'-.',c='lightsalmon',linewidth=3)

    else:
        # plt.scatter(results[:,0],results[:,1]*100,550,'dodgerblue',marker='.',label='Shape Ensemble')
        # plt.plot(results[:,0],results[:,2]*100,'-b',linewidth=1)
        plt.plot(results[:,0]*2,ysmoothed*100,':',c='dodgerblue',linewidth=3)
        
plt.xlabel('Equivalent Diameter / mm',fontsize=18)
plt.ylabel('Sensitivity / %',fontsize=18)

# raw_data = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                          # markersize=10, label='Raw Data')
# smoothed_regression2d = mlines.Line2D([], [], color='black', marker='none', linestyle='-.',
#                           linewidth=1.5, label='2D Smoothed Regression')
# smoothed_regression3d = mlines.Line2D([], [], color='black', marker='none', linestyle='--',
                          # linewidth=1.5, label='3D Smoothed Regression')
raw_or2d = mlines.Line2D([], [], color='mediumorchid', marker='none', linestyle='-.',
                          linewidth=2.2, label='2D OR')
# raw_or3d = mlines.Line2D([], [], color='rebeccapurple', marker='none', linestyle='--',
#                           linewidth=2.2, label='3D OR (2D Patch + Shape)')

rawp = mlines.Line2D([], [], color='indianred', marker='none', linestyle='-',
                          linewidth=2.2, label='Patch Model')
rawt = mlines.Line2D([], [], color='lightsalmon', marker='none', linestyle='-.',
                          linewidth=2.2, label='Tile Model')
raws = mlines.Line2D([], [], color='dodgerblue', marker='none', linestyle=':',
                          linewidth=2.2, label='Shape Ensemble')
# Doctor points: Unenhanced CT for the detection of renal cell carcinoma: effect of tumor size and contour type
doctor_spec = [98,98]
doctor_sens = [27,87]
doctor_size = [15,40]

plt.legend(handles=[rawp,rawt,raw_or2d,raws],ncol=4,fontsize=11,
           bbox_to_anchor=[0.95,-0.11])
plt.xlim(19,81)
plt.ylim(-2,102)
plt.show()
plt.savefig('/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/evaluate/inference_unseen/or_specsens_continuous.png',bbox_inches='tight')

