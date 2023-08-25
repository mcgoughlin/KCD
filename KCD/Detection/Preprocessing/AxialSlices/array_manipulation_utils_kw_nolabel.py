# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:23:52 2022

@author: mcgoug01
"""
import nibabel as nib
import os
import numpy as np
from numpy.lib import stride_tricks as st
import torch.nn as nn
import torch
import collections
from scipy import stats

def get(case_string,cases_folder,seg_path):
    case_path = os.path.join(cases_folder, "images", case_string)
    seg_fp = os.path.join(seg_path, case_string)
    return nib.load(case_path), nib.load(seg_fp)

def rescale_array(im, spacing,
                   target_spacing = np.array([3,3,3]), order=1,is_seg=False):

    if spacing is None:
        raise ValueError('spacing must be given as input when apply_resizing=True.')

    scale = np.array(spacing) / target_spacing
    target_shape = tuple(np.round(im.shape * scale).astype(int))
    if is_seg:
        return nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(im),dim=0),dim=0),
                                         mode = 'nearest', size =target_shape).numpy()[0,0]
    else:
        return nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(im),dim=0),dim=0),
                                         mode = 'trilinear', size =target_shape).numpy()[0,0]

def _get_xyz_list(shape,patch_size = np.array([32,32,32]),overlap=0.1,boundary_z=0):
    
    nz, nx, ny = shape
    overlap_arr = overlap * patch_size
    inplane = stats.mode(overlap_arr)[0]
    overlap_arr = np.where(overlap_arr==inplane,overlap_arr,boundary_z)

    n_patches = np.ceil((np.array([nz, nx, ny]) - patch_size) / 
                        overlap_arr).astype(int)
    if patch_size[-1] ==1:
        n_patches[-1] = ny
        
    for i in range(len(n_patches)):
        if n_patches[i] ==0:
            n_patches[i] =1
        
    # upper left corners of all patches
    z_list = np.linspace(0, nz - patch_size[0], n_patches[0]).astype(int).tolist()
    x_list = np.linspace(0, nx - patch_size[1], n_patches[1]).astype(int).tolist()
    y_list = np.linspace(0, ny - patch_size[2], n_patches[2]).astype(int).tolist()
    xyz_list = []
    for z in z_list:
        for x in x_list:
            for y in y_list:
                xyz_list.append((z, x, y))
                # print(z, x, y)
        # assert(1==2)
    return xyz_list

def extract_sliding_windows(arr, patch_size=(32,32,32),overlap=0.1,boundary_z=0):

    # in case the volume is smaller than the patch size we pad it
    # and save the input size to crop again before returning
    volume = torch.unsqueeze(torch.Tensor(arr),dim=0)
    shape_in = np.array(volume.shape)
    #  possible padding of too small volumes
    pad = [0, patch_size[1] - shape_in[2], 0, patch_size[2] - shape_in[3],
           0, patch_size[0] - shape_in[1]]
    pad = np.maximum(pad, 0).tolist()
    volume = nn.functional.pad(volume, pad).type(torch.float)
    shape = volume.shape[1:]

    
    #  get all top left coordinates of patches
    xyz_list = _get_xyz_list(shape,patch_size=patch_size,overlap=overlap,boundary_z=boundary_z)

    # introduce batch size
    # some people say that introducing a batch size at inference time makes it faster
    # I couldn't see that so far
    n_full_batches = len(xyz_list)
    xyz_batched = [xyz_list[i : (i + 1) ]
                   for i in range(n_full_batches)]
    
    if n_full_batches < len(xyz_list):
        xyz_batched.append(xyz_list[n_full_batches:])
        
    sliding_windows = torch.squeeze(torch.stack([volume[:,
                                pack[0][0]:pack[0][0]+patch_size[0],
                                pack[0][1]:pack[0][1]+patch_size[1],
                                pack[0][2]:pack[0][2]+patch_size[2]] for pack in xyz_batched])).numpy()
    
    # assert(1==2)
    print(sliding_windows.shape[-3:],patch_size)
    assert(sliding_windows.shape[-3:]==tuple(patch_size))
        
    return sliding_windows

def count_elements(array):
    vals,counts = np.unique(array,return_counts=True)
    kid_count,tum_count = 0,0
    if 1 in vals: kid_count = counts[vals==1][0]
    if 2 in vals: tum_count = counts[vals==2][0]
    return kid_count+tum_count, tum_count

def extract_labels(sw_seg, kidney_vol_thresh = None):
    
    labels = sw_seg.reshape(sw_seg.shape[0],-1)
    counts = np.apply_along_axis(count_elements,-1,labels)
    # return counts
    return np.array([1 if kids>kidney_vol_thresh else 0 for kids,tums in counts])
    # return np.array([2 if ((tums>tumour_vol_thresh)) else 1 if kids>kidney_vol_thresh else 0 for kids,tums in counts])


    