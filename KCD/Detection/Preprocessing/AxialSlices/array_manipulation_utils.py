# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:23:52 2022

@author: mcgoug01
"""
import nibabel as nib
import os
import numpy as np
import torch.nn as nn
import torch
import collections
from scipy import stats
import file_utils as fu
import SimpleITK as sitk
from skimage.measure import regionprops

def get(case_string,im_path,seg_path):
    case_path = os.path.join(im_path, case_string)
    seg_path = os.path.join(seg_path, case_string)
    
    return nib.load(case_path), nib.load(seg_path)

def rescale_array(im, spacing,axes,
                   target_spacing = np.array([3,3,3]), order=1,is_seg=False):
    if spacing is None:
        raise ValueError('spacing must be given as input when apply_resizing=True.')

    scale = np.array(spacing) / target_spacing[axes]
    target_shape = np.round(np.array(im.shape)[axes] * scale).astype(int)
    target_shape = tuple(target_shape[np.array((np.argmax(axes==0),np.argmax(axes==1),np.argmax(axes==2)))])
    
    if is_seg:
        return nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(im),dim=0),dim=0),
                                         mode = 'nearest', size =target_shape).numpy()[0,0]
    else:
        return nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(im),dim=0),dim=0),
                                         mode = 'trilinear', size =target_shape).numpy()[0,0]

def filter_shifted_windows(sw_im,sw_seg,canc_thresh,kid_thresh,target_spacing,
                           shuffle=True,has_seg_label=True):
    
    labels = extract_labels(sw_seg,tumour_vol_thresh = canc_thresh, kidney_vol_thresh = kid_thresh,has_seg_label=has_seg_label)
    maligs = sw_im[labels==2]
    normals = sw_im[labels==1]
    none = sw_im[labels==0]
    filter_out_blank = none.max(axis=-1).max(axis=-1).max(axis=-1)>-200
    none = none[filter_out_blank]

    if len(maligs)>0: maligs = maligs[maligs.reshape(len(maligs),-1).max(axis=1)>0]
    if len(normals)>0: normals = normals[normals.reshape(len(normals),-1).max(axis=1)>0]

    # shuffle order so the save limit does not bias saved patches to the top portions of the image
    if shuffle:
        for arr in [maligs,normals,none]:np.random.shuffle(arr)
    return maligs,normals,none

def count_elements(array):
    vals,counts = np.unique(array,return_counts=True)
    kid_count,tum_count = 0,0
    if 1 in vals: kid_count = counts[vals==1][0]
    if 2 in vals: tum_count = counts[vals==2][0]
    return kid_count+tum_count, tum_count

def extract_labels(sw_seg, has_seg_label=True,
                   tumour_vol_thresh = None, kidney_vol_thresh = None):
    
    labels = sw_seg.reshape(sw_seg.shape[0],-1)
    counts = np.apply_along_axis(count_elements,-1,labels)
    if has_seg_label:return np.array([2 if ((tums>tumour_vol_thresh) and (kids>kidney_vol_thresh)) else 1 if kids>kidney_vol_thresh else 0 for kids,tums in counts])
    else:return np.array([1 if kids>kidney_vol_thresh else 0 for kids,tums in counts])
    
def nifti_2_correctarr(im_n):
    aff = im_n.affine
    im = sitk.GetImageFromArray(im_n.get_fdata())
    im.SetOrigin(-aff[:3,3])
    im.SetSpacing(im_n.header['pixdim'][1:4].tolist())
    
    ##flips image along correct axis according to image properties
    flip_im = sitk.Flip(im, np.diag(aff[:3,:3]<-0).tolist())
    nda = np.rot90(sitk.GetArrayViewFromImage(flip_im))
    return nda.copy()

def get_spacing(im_nib):
    return np.abs(im_nib.header['pixdim'][1:4])

# def find_orientation(spacing,kidney_centroids):
#     rounded_spacing = np.around(spacing,decimals=2)
#     assert(2 in np.unique(rounded_spacing,return_counts=True)[1])
#     inplane_spacing = stats.mode(rounded_spacing,keepdims=False)[0]
#     indices = np.array([0,1,2])
#     axial = indices[rounded_spacing!=inplane_spacing][0]
    
#     if len(kidney_centroids)==1:
#         return axial,*indices[rounded_spacing==inplane_spacing]

#     kidney_differences = np.abs(np.array(kidney_centroids[0]) - np.array(kidney_centroids[1]))
#     kidney_differences[axial]=0
#     leftright = indices[np.argmax(kidney_differences)]
#     updown = indices[(indices!=axial) & (indices!=leftright)][0]
#     return axial,leftright,updown

def find_orientation(spacing,is_axes=True,im=None):
    rounded_spacing = np.around(spacing,decimals=1)

    if not (2 in np.unique(rounded_spacing,return_counts=True)[1]): return 0,0,0
    inplane_spacing = stats.mode(rounded_spacing,keepdims=False)[0]
    indices = np.array([0,1,2])
    axial = indices[rounded_spacing!=inplane_spacing][0]
    if is_axes:
        if axial==0: 
            first_half = im[:,:256]
            second_half = im[:,256:]
        else:
            first_half = im[:256]
            second_half = im[256:]
            
        # there should only be one (rough) plane of symmetry in a CT scan: 
        # from the axial perspective that splits the spine in half - leftright when facing person. 
        # Thus, the symmetrical 
        # plane for bones should be up-down. We know the axial plane index already, so to determine 
        # up-down, we simplysplit image along the first non-axial dimension and in half,
        # and compare bone 
        # totals in each half. if these are roughly similar (within 30% of each other) - we 
        # say this is symmetry, and therefore the up-down plane.
        try:first_total = np.array(regionprops((first_half>250).astype(int))[0].area)
        except(IndexError):first_total=0 # index error occurs when not a single bit of bone occurs
        try:second_total = np.array(regionprops((second_half>250).astype(int))[0].area)
        except(IndexError):second_total=0
        fraction = first_total/(second_total+1e-6)
        if (fraction>0.7) and (fraction < 1.3):
            if axial==0: lr,ud, = 1,2
            elif axial ==1:lr,ud=0,2
            else: lr,ud=0,1
        else:
            if axial==0: lr,ud, = 2,1
            elif axial ==1:lr,ud=2,0
            else: lr,ud=1,0

        return axial,lr,ud
    # do below when assigning orientations for spacings - where lr and ud distinction doesnt matter
    else:return axial,*indices[rounded_spacing==inplane_spacing][::-1]

    