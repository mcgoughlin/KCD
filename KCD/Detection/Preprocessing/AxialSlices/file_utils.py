#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:22:56 2023

@author: mcgoug01
"""
import os

def create_save_path_structure(path,save_dir,thresh,data_name=None):
    assert(os.path.exists(save_dir))
    save_path = os.path.join(save_dir,"test")
    if not os.path.exists(save_path): os.mkdir(save_path)
    assert( not (data_name == None))
    save_path = os.path.join(save_path,data_name)
    if not os.path.exists(save_path): os.mkdir(save_path)
        
    if not os.path.exists(save_path): os.mkdir(save_path)

    
    return save_path

def filename_structure(path,name,voxel_size_mm,cancthresh,kidthresh,
                       depth_z,boundary_z,dilate,has_seg_label=True):
    
    if not os.path.exists(path): os.mkdir(path)
    
    fold = os.path.join(path,"Voxel-"+str(voxel_size_mm)+"mm-Dilation"+str(dilate)+"mm")
    if not os.path.exists(fold): os.mkdir(fold)
    
    fold = os.path.join(fold,"CancerThreshold-"+str(cancthresh)+'mm-KidneyThreshold'+str(kidthresh)+'mm')
    if not os.path.exists(fold): os.mkdir(fold)
    
    if depth_z>1:
        fold = os.path.join(fold,"3D-Depth"+str(depth_z)+'mm-Boundary'+str(boundary_z)+'mm')
        if not os.path.exists(fold): os.mkdir(fold)
    else:
        fold = os.path.join(fold,"2D-Boundary"+str(boundary_z)+'mm')
        if not os.path.exists(fold): os.mkdir(fold)
        
    if has_seg_label:
        fold = os.path.join(fold,"labelled_data")
        if not os.path.exists(fold): os.mkdir(fold)
    else:
        fold = os.path.join(fold,"unseen_test_data")
        if not os.path.exists(fold): os.mkdir(fold)
        
    fold = os.path.join(fold,name)
    if not os.path.exists(fold): os.mkdir(fold)
    
    return fold
