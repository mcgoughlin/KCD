#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:22:56 2023

@author: mcgoug01
"""
import os

def create_save_path_structure(path,save_dir,data_name=None):
    assert(os.path.exists(save_dir))
    save_path = os.path.join(save_dir,"slices")
    if not os.path.exists(save_path): os.mkdir(save_path)
    assert( not (data_name == None))
    save_path = os.path.join(save_path,data_name)
    if not os.path.exists(save_path): os.mkdir(save_path)
    
    return save_path


def create_save_path_structure_seg(path, save_dir, data_name=None):
    assert (os.path.exists(save_dir))
    save_paths = []
    for pathtype in ['segs','slices']:
        save_path = os.path.join(save_dir, pathtype)
        if not os.path.exists(save_path): os.mkdir(save_path)
        assert (not (data_name == None))
        save_path = os.path.join(save_path, data_name)
        if not os.path.exists(save_path): os.mkdir(save_path)

        save_paths.append(save_path)

    return save_paths

def filename_structure_labelled(path,name,voxel_size_mm,cancthresh,kidthresh,
                       depth_z,boundary_z,dilate):
    
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
        
    fold = os.path.join(fold,"labelled_data")
    if not os.path.exists(fold): os.mkdir(fold)

        
    fold = os.path.join(fold,name)
    if not os.path.exists(fold): os.mkdir(fold)
    
    return fold

def filename_structure_unlabelled(path,name,voxel_size_mm,foreground_thresh,
                       depth_z,boundary_z,dilate):
    
    if not os.path.exists(path): os.mkdir(path)
    
    fold = os.path.join(path,"Voxel-"+str(voxel_size_mm)+"mm-Dilation"+str(dilate)+"mm")
    if not os.path.exists(fold): os.mkdir(fold)
    
    fold = os.path.join(fold,"ForegroundThresh-"+str(foreground_thresh)+'mm')        
    if not os.path.exists(fold): os.mkdir(fold)
    
    if depth_z>1:
        fold = os.path.join(fold,"3D-Depth"+str(depth_z)+'mm-Boundary'+str(boundary_z)+'mm')
        if not os.path.exists(fold): os.mkdir(fold)
    else:
        fold = os.path.join(fold,"2D-Boundary"+str(boundary_z)+'mm')
        if not os.path.exists(fold): os.mkdir(fold)

    fold = os.path.join(fold,"unlabelled_data")
    if not os.path.exists(fold): os.mkdir(fold)
        
    fold = os.path.join(fold,name)
    if not os.path.exists(fold): os.mkdir(fold)
    
    return fold
