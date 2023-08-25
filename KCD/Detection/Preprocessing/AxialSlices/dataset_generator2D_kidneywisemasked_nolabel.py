# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:21:45 2022

@author: mcgoug01
"""

from array_manipulation_utils_2dkw_nolabel import *
import numpy as np
import os
import SimpleITK as sitk
from skimage.measure import regionprops
import scipy.ndimage as spim
import cv2
import torch.nn as nn
import torch

def filename_structure(path,name,voxel_size_mm,thresh):
    if not os.path.exists(path): os.mkdir(path)
    
    fold = os.path.join(path,"Voxel-"+str(voxel_size_mm)+"mm")
    if not os.path.exists(fold): os.mkdir(fold)
    
    fold = os.path.join(fold,"Threshold-"+str(thresh)+'mm')
    if not os.path.exists(fold): os.mkdir(fold)
    
    fold = os.path.join(fold,name)
    if not os.path.exists(fold): os.mkdir(fold)
    
    return fold

def nifti_2_correctarr(im_n):
    aff = im_n.affine
    im = sitk.GetImageFromArray(im_n.get_fdata())
    im.SetOrigin(-aff[:3,3])
    im.SetSpacing(im_n.header['pixdim'][1:4].tolist())
    
    ##flips image along correct axis according to image properties
    flip_im = sitk.Flip(im, np.diag(aff[:3,:3]<-0).tolist())
    
    
    nda = np.rot90(sitk.GetArrayViewFromImage(flip_im))
    return nda.copy()

def get_masses(binary_arr,vol_thresh,intensity_image = None):
    return [[mass,mass.centroid] for mass in regionprops(spim.label(binary_arr)[0],intensity_image = intensity_image) if mass.area>vol_thresh]

def is_sole_kidney_central(kidney_centroids, im,inf, inplane_spac,
                           test1_length = 25, test2_length = 10):
    sole_kidney = kidney_centroids[0][1]
    
    # Find centre of bone-attenuating tissue - lr is left-to-right on axial, ud is up-down
    ud_bone,lr_bone,z_bone = regionprops((im>250).astype(int))[0].centroid
    
    # test 1 - does the centre of the single kidney line up with spine within 25mm? if so - central kidney
    if abs(sole_kidney[1] - lr_bone)*inplane_spac < test1_length:
        return True, ud_bone, lr_bone
    else:
        # test 2 - kidney is also central if wraps around spine.
        # does some portion of the kidney wrap around the spine?
        
        # test 2 distance is 10mm
        _test_extent_inpixels = int((test2_length/2)*inplane_spac)
        
        # create test label - where the pixels within +-10mm of centre of bone attenuating tissue are zeroed out
        _central_test = inf
        _central_test[:,
                      int(lr_bone-_test_extent_inpixels):int(lr_bone+_test_extent_inpixels),
                      :] = 0
        
        # wrapping is true if one or more objects from test label appear either side of the centre of bone attenuating tissue.
        # if wrapping is true, then the kidney is central.
        _test_centroids = [ centroid[1]> lr_bone for _, centroid in get_masses(_central_test,0)]
        if (False in _test_centroids) and (True in _test_centroids):
            return True, ud_bone, lr_bone
        else:
            return False, ud_bone, lr_bone
        
    
def create_save_path_structure(path,save_dir,thresh,data_name=None):
    assert(os.path.exists(path))
    assert( not (data_name == None))
    if data_name == None:
        data_path = os.path.join(path,"raw_data")
    else:
        data_path = os.path.join(path,"raw_data",data_name)
        
    print(data_path)
    assert(os.path.exists(data_path))
    
    
    save_path = os.path.join(save_dir,"sliding_window_kidneywisemasked")
    if not os.path.exists(save_path): os.mkdir(save_path)
    assert( not (data_name == None))
    save_path = os.path.join(save_path,data_name)
    if not os.path.exists(save_path): os.mkdir(save_path)
        
    if not os.path.exists(save_path): os.mkdir(save_path)

    
    return save_path,data_path
    
def get_shifted_windows(reshaped_im,reshaped_seg,
                        overlap=None,patch_size=None):
    

    sw_reshape_im = extract_sliding_windows(reshaped_im, overlap=overlap, patch_size=patch_size)
    sw_reshape_seg = extract_sliding_windows(reshaped_seg, overlap=overlap, patch_size=patch_size)  
    
    return sw_reshape_im, sw_reshape_seg

def get_centralised_windows(reshaped_im,reshaped_seg,centroid,patch_size=None,is_kits=True):
    print("is_kits:",is_kits,patch_size)
    shape_in = reshaped_im.shape
    xc,zc,yc = centroid

    pad = [0, patch_size[1] - shape_in[1], 0, patch_size[2] - shape_in[2],
           0, patch_size[0] - shape_in[0]]
    pad = np.maximum(pad, 0).tolist()
    reshaped_im = nn.functional.pad(torch.Tensor(reshaped_im), pad).type(torch.float).numpy()
    reshaped_seg = nn.functional.pad(torch.Tensor(reshaped_seg), pad).type(torch.float).numpy()
    shape = reshaped_im.shape
    print(shape)
    

    if is_kits:
        top_left_x = max(0,int(centroid[0]-(patch_size[0]/2)))
        top_left_y = max(0,int(centroid[2]-(patch_size[2]/2)))
        bottom_right_y = min(shape[2],top_left_y+patch_size[0])
        bottom_right_x = min(shape[0],top_left_x+patch_size[0])
        
        if bottom_right_x == shape[0]: top_left_x = bottom_right_x-patch_size[0]
        if bottom_right_y == shape[2]: top_left_y = bottom_right_y-patch_size[2]
        central_window_im = np.array([reshaped_im[top_left_x:bottom_right_x,
                                          i,
                                          top_left_y:bottom_right_y] for i in range(shape[1])])
        
        central_window_seg = np.array([reshaped_seg[top_left_x:bottom_right_x,
                                          i,
                                          top_left_y:bottom_right_y] for i in range(shape[1])])
    else:
        top_left_x = max(0,int(centroid[0]-(patch_size[0]/2)))
        top_left_y = max(0,int(centroid[1]-(patch_size[1]/2)))
        bottom_right_y = min(shape[1],top_left_y+patch_size[0])
        bottom_right_x = min(shape[0],top_left_x+patch_size[0])
        
        if bottom_right_x == shape[0]: top_left_x = bottom_right_x-patch_size[0]
        if bottom_right_y == shape[1]: top_left_y = bottom_right_y-patch_size[1]
        central_window_im = np.array([reshaped_im[top_left_x:bottom_right_x,                                
                                          top_left_y:bottom_right_y,
                                          i] for i in range(shape[2])])
        
        central_window_seg = np.array([reshaped_seg[top_left_x:bottom_right_x,
                                          top_left_y:bottom_right_y,
                                          i] for i in range(shape[2])])
        
    print(top_left_x,bottom_right_x,top_left_y,bottom_right_y)

    return central_window_im, central_window_seg

def filter_shifted_windows(sw_im,sw_seg,kid_thresh,target_spacing,shuffle=True):
    
    labels = extract_labels(sw_seg,kidney_vol_thresh = kid_thresh)
    maligs = sw_im[labels==2]
    normals = sw_im[labels==1]
    none = sw_im[labels==0]
    none = none[none.max(axis=-1).max(axis=-1)>-200]
    
    if len(maligs)>0: maligs = maligs[maligs.reshape(len(maligs),-1).max(axis=1)>0]
    if len(normals)>0: normals = normals[normals.reshape(len(normals),-1).max(axis=1)>0]

    # shuffle order so the save limit does not bias saved patches to the top portions of the image
    if shuffle:
        for arr in [maligs,normals,none]:
            np.random.shuffle(arr)
    return maligs,normals,none


def save_windows(windowed_im,windowed_seg,
                 kid_thresh,target_spacing,save_path,voxel_size_mm,
                 thresh_r_mm,case_name,kidney_side='random',
                 shuffle=True,save_limit=100,centralised=False):
    
    if centralised: string = 'centralised'
    else: string = 'shifted'
    
    sw_maligs,sw_normals,sw_none= filter_shifted_windows(windowed_im,windowed_seg,kid_thresh,target_spacing[0],shuffle=shuffle)

    sw_subtypes = [sw_maligs,sw_normals,sw_none]
    names = ["tumour","kidney","none"]
    if centralised:
        names.pop(-1)
        sw_subtypes.pop(-1)
        sw_maligs,sw_normals,sw_none= filter_shifted_windows(windowed_im,windowed_seg,kid_thresh,target_spacing[0],shuffle=shuffle)
        print("Creating {} tumour, and {} kidney {} windows.".format(min(len(sw_maligs),save_limit),
                                                                          min(len(sw_normals),save_limit),
                                                                          string))
    else:
        print("Creating {} none, {} tumour, and {} kidney {} windows.".format(min(len(sw_none),save_limit),
                                                                              min(len(sw_maligs),save_limit),
                                                                          min(len(sw_normals),save_limit),
                                                                          string))
    for subname, label_group in zip(names,sw_subtypes):
        fold = filename_structure(save_path,subname,voxel_size_mm,thresh_r_mm)
        for i,sw in enumerate(label_group):
            if i >= save_limit: break
            sw = np.expand_dims(np.expand_dims(sw,axis=0),axis=0)
            np.save(os.path.join(fold,"{}_{}_{}_index{}".format(case_name[:-7],kidney_side,string,i)),sw)
            
def get_kid_str(kidney_data,reshaped_im,reshaped_seg,spacing,is_kits=True):
    index = 1
    
    if len(kidney_data)==1:
        central_kidney_flag, ud_bone, lr_bone = is_sole_kidney_central(kidney_data,reshaped_im,
                                                                       reshaped_seg,np.mean(spacing[1:]))
        if central_kidney_flag:
            kid_str = ['central']
        elif kidney_data[0][1][index] - lr_bone > 0:
            kid_str = ['left']
        else:
            kid_str = ['right']
            
        print("sole kidney is {}".format(kid_str))
    elif len(kidney_data) == 0: assert(1==2)
    elif len(kidney_data)>2: 
        kid_str=['_failure']
    else:
        first_kidney = kidney_data[0]
        second_kidney = kidney_data[1]
        if first_kidney[1][index] < second_kidney[1][index]:
            kid_str = ['right','left']
        else:
            kid_str = ['left','right']
            
    return kid_str

def get_spacing(im_nib):
    spacing = np.abs(im_nib.header['pixdim'][1:4])
    temp = spacing[1]
    spacing[1]=spacing[0]
    spacing[0]=temp
    return spacing
    
def create_dataset(int_list,im_path,seg_path,target_spacing,overlap,
                   patch_dims,thresh_r_mm,save_path,
                   voxel_size_mm, save_limit,is_3D=True,bbox_boundary_mm=50):
    bbox_boundary = int(np.round(bbox_boundary_mm/target_spacing[0]))
    vol_thresh_vox = 1000 / float(np.prod(target_spacing)) # ignore 'kidney' segmentations with a vol less than 100mm cubed
    kernel = spim.generate_binary_structure(3, 2).astype(np.uint8)
    
    for get_int in int_list:
        if get_int=='.DS_Store':continue
        if get_int=='KiTS-00151.nii.gz': continue #skip this case - strange label artefact
        ct,seg = get(get_int,im_path,seg_path)
        ct_im,seg_im = nifti_2_correctarr(ct), nifti_2_correctarr(seg)
        patch_size= patch_dims.copy()
        print(get_int, patch_size,patch_dims)
        spacing=get_spacing(ct)  
        
        if get_int.startswith('KiTS'):is_kits=True
        else:is_kits=False
        
        if is_kits:
            patch_size[1] = 1
        else:
            patch_size[2] = 1
        
        print(patch_size)
            
        reshaped_im = rescale_array(ct_im,spacing,target_spacing = target_spacing)
        reshaped_seg = rescale_array(seg_im,spacing,target_spacing = target_spacing,is_seg=True)
        kidney_data = np.asarray([[mass,centroid] for mass, centroid in get_masses(reshaped_seg>0,vol_thresh_vox)],dtype=object)

        kid_str = get_kid_str(kidney_data,reshaped_im,reshaped_seg,spacing,is_kits=is_kits)
        if kid_str == ['_failure']:continue
        kid_thresh = ((10/target_spacing[0])**2) * 3.1416
        

                
        for kidney_datum, name in zip(kidney_data,kid_str):
            print("\nGenerating from {} {}-side.".format(get_int,name))
            #  this data is for training only - testing will ignore generic 
            x1,y1,z1,x2,y2,z2 = kidney_datum[0].bbox
            centroid = kidney_datum[1]
            
            ### final mask should be generated from a 3 dilation of the masked segmentation.
            first_mask = np.zeros_like(reshaped_seg)
            first_mask[x1:x2,y1:y2,z1:z2] = np.ones((x2-x1,y2-y1,z2-z1))
            first_mask = first_mask*(reshaped_seg>0).astype(np.uint8)
            final_mask = spim.binary_dilation(first_mask, kernel, iterations=bbox_boundary)
            
            seg = first_mask*reshaped_seg
            ct = final_mask*reshaped_im

            # get rid of zero'd background that confounds training - make bg -200HU, the min possible val in image.
            ct += np.where(final_mask==0,-200,0)
            
            # sw_im, sw_seg = get_shifted_windows(ct,seg,overlap=overlap,patch_size=patch_size)  
            
            # assert(sw_im.shape[-2:]==(224,224))
            # assert(sw_seg.shape[-2:]==(224,224))
                       
            # save_windows(sw_im,sw_seg,kid_thresh,
            #              target_spacing,save_path,
            #              voxel_size_mm,thresh_r_mm,get_int,
            #              shuffle=True,save_limit=save_limit,centralised=False)

            cent_im, cent_seg = get_centralised_windows(ct,seg,centroid,patch_size=patch_size,is_kits=is_kits) 
            
            print(cent_im.shape)
            assert(cent_im.shape[-2:]==(224,224))
            assert(cent_seg.shape[-2:]==(224,224))
            
            save_windows(cent_im,cent_seg,kid_thresh,
                         target_spacing,save_path,
                         voxel_size_mm,thresh_r_mm,get_int,kidney_side=name,
                         shuffle=False,save_limit=1e4,centralised=True)

                
                
def create_2D(im_path,seg_path, voxel_size_mm,
              overlap_mm, patch_voxels, save_dir,
              thresh_r_mm = 7.5,save_limit=1000,
              train_fraction=0.8,data_name = False,bbox_boundary_mm=50):
        
    target_spacing = np.array([voxel_size_mm]*3)
    patch_size = np.array([patch_voxels]*3)
    overlap = overlap_mm/(patch_voxels*voxel_size_mm)
    save_path,data_path = create_save_path_structure(im_path,data_name=data_name,save_dir=save_dir,thresh=thresh_r_mm)    
    case_integers = [file for file in os.listdir(os.path.join(data_path,"images"))]
    

    print("Creating Training Dataset")
    create_dataset(case_integers,data_path,seg_path,target_spacing,overlap,
                   patch_size,thresh_r_mm,save_path,
                   voxel_size_mm,save_limit,is_3D=False,
                   bbox_boundary_mm=bbox_boundary_mm)

if __name__ == "__main__":
    path = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data"
    save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/CNN_dataset'
    segpath='/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/add_ncct_unseen/[4 4 4]mm'
    overlap_mm = 40## this dictates the minimum distance apart from each window! not the overlap.
    patch2d = 224
    save_limit_percase_perlabel = 100
    voxel_spacings = [1]
    thresholds_r_mm = [0]
    for thresh in thresholds_r_mm:
        for spacing in voxel_spacings: 
            # overlap_mm = spacing*patch2d/10
            print("Voxel Spacing {}mm".format(spacing))
            create_2D(path,segpath,spacing,overlap_mm,patch2d,save_dir,data_name='add_ncct_unseen',
                      thresh_r_mm=thresh, save_limit=save_limit_percase_perlabel,bbox_boundary_mm=40)