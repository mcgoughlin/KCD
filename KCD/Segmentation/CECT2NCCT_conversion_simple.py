# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:35:39 2022

@author: mcgoug01
"""


from os.path import *
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage.morphology import binary_erosion, binary_dilation

# CECT_path = "/Users/mcgoug01/Downloads/AbCT1k/all_images"
# seg_path = "/Users/mcgoug01/Downloads/AbCT1k/all_labels"

CECT_path = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/kits23/all_images"
seg_path = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/kits23/all_labels"

import torch.nn as nn
pool = nn.AvgPool3d(7,stride=1,padding=3).cuda()

### Philosophy:
#   Convert CECT to NCCT based on attenutations and segmentations provided by 
#   radiologists.

def get_ims(integer):

    case = "case_{:05d}.nii.gz".format(integer)
    impath = join(CECT_path,case)
    segpath = join(seg_path,case)
    
    return (nib.load(impath).get_fdata(), nib.load(segpath).get_fdata(), nib.load(impath).affine)
    
def diff_shower(integer,slice= 60):
    #seg is segmentation of soft tissue
    im,seg,aff = get_ims(integer)
    
    seg_bin = (seg>0).astype(int)
    overextended_seg = binary_dilation(seg_bin)
    
    #highlight soft tissue
    create_diff = create_label(im)
    
    #create a non-contrast attenuation distribution for kidney and soft tissue across body
    lab_all_soft = np.random.normal(loc=57.5, scale=27, size = seg.shape)
    lab_kid = np.random.normal(loc=32, scale=15, size = seg.shape)
    
    #remove all non-segmented areas in the new distributions
    new_soft = np.multiply(lab_all_soft,create_diff).astype(np.float16)
    new_kid = np.multiply(lab_kid,overextended_seg).astype(np.float16)
        
    diff = np.where(create_diff>0,new_soft,im)
    sNCCT = np.where(overextended_seg>0,new_kid,diff)
    sNCCT_fat_reinstated = np.where(im>20,sNCCT,im)

    return sNCCT_fat_reinstated,aff,seg
## highlights areas that are soft tissue likely to be affected by contrast

## from paper: https://click.endnote.com/viewer?doi=10.1148%2Fradiol.09082074&token=WzM1MDQ0MjQsIjEwLjExNDgvcmFkaW9sLjA5MDgyMDc0Il0.lwQcElf-y2-Um9KdcQQEoFAjSZc
## mean (stdev) attenuation of the cortex in CECT:  211 +-20

## other paper had following CECT values for cortex
## mean (stdev) attenuation of the cortex in CECT:  147 +-40

## mean (stdev) attenuation of the medulla in UECT:  ?
##                                           CECT:  129 +-19

## parenchyma = cortex (outside)+medulla(inside)
## mean (stdev) attenuation of the parenchyma in UECT:  32+-15

## average liver intensity = https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4854642/
## spleen roughly 55
## liver roughly 59 HU

#no mention of standard deviations, assume same ratio as kid:liver mean
# this makes standrd dev of other = 15 * (57.5/32) = 27

def create_label(im):
    #3.29 corresponds to 3.29 standardd deviations
    lo = 129 - 3.29*19
    hi = 129 + 3.29*19
    
    #soft tissue only - cues taken from kidney tissue
    im_filtered = np.where((im>lo) & (im<hi),im,0)

                        
    #remove label extremes
    # lb = np.where(lb<20,0,lb)
    # return lb
    lb_thresh = im_filtered>0
    # return np.where(dilate==1,lb,0)
    return lb_thresh
    
save_path = "/Users/mcgoug01/Downloads/all_sncct/"
from os import *
for case in listdir(CECT_path):
    im_sv = join(save_path,"images")
    lb_sv = join(save_path,"labels")
    if exists(join(lb_sv,case)):continue

    int_case = int(case.split('.')[0][-5:])
    if int_case<300:continue

    if exists(join(lb_sv,case)):continue
    print(case,int_case)
    try:
        sNCCT,aff,seg = diff_shower(int_case)
    except: continue
    seg = np.round(seg)

    nib_im = nib.Nifti1Image(sNCCT.astype(np.int16),aff)
    nib.save(nib_im,join(im_sv,case))
    
    nib_lb = nib.Nifti1Image(seg.astype(np.int16),aff)
    nib.save(nib_lb,join(lb_sv,case))