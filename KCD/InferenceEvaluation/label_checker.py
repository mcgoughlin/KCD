#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:31:10 2023

@author: mcgoug01
"""

import os
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/CNN_dataset/sliding_window_kidneywisemasked/add_ncct_unseen/Voxel-1mm/Threshold-0mm/kidney-kthresh10'
# path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/CNN_dataset/sliding_window_kidneywisemasked/add_ncct_unseen/Voxel-1mm/Threshold-0mm/kidney'

cases = [int(file.split('_')[1]) for file in os.listdir(path) if file.endswith('.npy')]
cases = np.unique(cases)
upto = 0
cases.sort()
for case in cases:
    if case<upto:continue
    for side in ['left','right']:
        print(case,side)
        try:
            fp = os.path.join(path, [file for file in os.listdir(path) if file.startswith('RCC_{:03d}_{}_centralised'.format(case,side))][0])
        except:
            print('no files')
            continue
        im = np.load(fp)[0]
        # plt.subplot(121)
        plt.imshow(im[0],vmin=-200,vmax=200)
        # plt.subplot(122)
        # plt.imshow(im[15],vmin=-200,vmax=200)
        plt.show(block=True)
        