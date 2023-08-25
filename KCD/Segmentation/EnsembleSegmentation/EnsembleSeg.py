# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:12:09 2022

@author: mcgoug01
"""

from scipy import ndimage 

import torch
import nibabel as nib
import torch.nn as nn
from os import *
environ['OV_DATA_BASE'] = '/home/wcm23/rds/hpc-work/add_ncct_unseen'
from os.path import *
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

from time import sleep
import sys
import gc
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x

from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
import SimpleITK as sitk
from codebase import running_inferseg
SegLoader, Segment, SegProcess = running_inferseg.get_3d_UNet, running_inferseg.SlidingWindowPrediction, running_inferseg.SegmentationPostprocessing


# from codebase.running_inferseg import get_3d_UNet as SegLoader
# from codebase.running_inferseg import SlidingWindowPrediction as Segment
# from codebase.running_inferseg import SegmentationPostprocessing as SegProcess
# from codebase.setup_crossfold_validation import setup_cv_folders as TileFolderSetup
# from codebase.setup_crossfold_validation import data_generation as TileGenerator

from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.data.SegmentationData import SegmentationData
from ovseg.utils.io import load_pkl, read_nii,_has_z_first
from ovseg.utils.torch_np_utils import maybe_add_channel_dim

class Ensemble_Seg(nn.Module):
    def __init__(self,data_name:str=None, ##seg preprocess args
                 seg_fp:str=None,spacing = np.array([3,3,3]),
                 seg_dev:str=None,do_prep=False,do_infer=False): ##seg args
        super().__init__()
        
        
        print("")
        print("Initialising Ensemble Segmentation System...")
        print("")
        torch.cuda.empty_cache()
        gc.collect()
        case_path = join(environ['OV_DATA_BASE'],'raw_data',data_name,'images')
        self.cases = [file for file in listdir(case_path) if file.endswith('.nii.gz') or file.endswith('.nii')]
        self.segpredpath =environ['OV_DATA_BASE']
        self.data_name = data_name
        
        
        # ### SEG PREPROCESS ###
        self.seg_dev = seg_dev
        self.preprocessed_name = str(spacing[0])+','+str(spacing[1])+','+str(spacing[2])+"mm"
        self.spacing =spacing
        
        self.seg_save_loc = join(self.segpredpath,"predictions_nii")
        self.seg_save_loc_lr = join(self.segpredpath,"predictions_npy")
        
        # all predictions are stored in the designated 'predictions' folder in the OV_DATA_BASE
        
        # creating folders 
        if not exists(self.seg_save_loc):
            makedirs(self.seg_save_loc)
        if not exists(self.seg_save_loc_lr):
            makedirs(self.seg_save_loc_lr)
            
        sv_fold = join(self.seg_save_loc,self.data_name)
        if not exists(sv_fold):
            mkdir(sv_fold)
                  
        lrsv_fold = join(self.seg_save_loc_lr,self.data_name)
        if not exists(lrsv_fold):
            mkdir(lrsv_fold)
            
        self.lrsv_fold_size = join(lrsv_fold,"{}mm".format(self.spacing))
        if not exists(self.lrsv_fold_size):
            mkdir(self.lrsv_fold_size)
            
        self.sv_fold_size = join(self.seg_save_loc,self.data_name,'{}mm'.format(self.spacing))
        if not exists(self.sv_fold_size):
            mkdir(self.sv_fold_size)
            
            
            
        if do_prep:
            self.Segmentation_Preparation(self.spacing,data_name = self.data_name)
        
        ### SEG ####
        print("Conducting Segmentation.")
        self.seg_mp_low = seg_fp
        if do_infer:
            self.Segment_CT()
        print("Segmentation complete!")
        print("")
        torch.cuda.empty_cache()
        gc.collect()
        

        
        
    def Segmentation_Preparation(self,seg_spacing,
                                 data_name = 'test'):
        

        pp_save_path = join(self.segpredpath,"preprocessed",self.data_name,self.preprocessed_name,'images')
        
        rawdata_path = join(self.segpredpath,"raw_data",self.data_name,'images')
        
        print("##SEG PREPROCESS##\nPreprocessing CT Volumes to {}\n Stored in location {}.".format(seg_spacing,pp_save_path))
        print("")
        preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                                  apply_pooling=False,
                                                  apply_windowing=True,
                                                  target_spacing=seg_spacing,
                                                  pooling_stride=None,
                                                  window=np.array([-116., 130.]),
                                                  scaling=np.array([41.301857, 12.257426]),
                                                  lb_classes=None,
                                                  reduce_lb_to_single_class=True,
                                                  lb_min_vol=None,
                                                  prev_stages=[],
                                                  save_only_fg_scans=False,
                                                  n_im_channels = 1)

        preprocessing.preprocess_raw_data(raw_data=data_name,
                                          preprocessed_name=self.preprocessed_name,
                                          data_name=None,
                                          save_as_fp16=True)
        print("")
        
        
    def _load_UNet(self, path = None,
                   dev=None):
        model_files = [file for file in listdir(path) if "fold_" in file]
        
        for foldpath in model_files:
            self.SegModel = SegLoader(1, 2, 6, 2, filters=32,filters_max=1024)
            b = 64
            sm = torch.load(join(path,foldpath,"network_weights"),map_location='cpu')
            self.SegModel.load_state_dict(sm)
            self.SegModel.to(self.seg_dev)
            self.SegModel.eval()
                
            self.Segment.append(Segment(self.SegModel,[b,b,b],batch_size=4,overlap=0.5,dev=self.seg_dev))

        
    def seg_pred(self, data_tpl,do_postprocessing=True):

        im = data_tpl['image']
        im = maybe_add_channel_dim(im)
        
        im = torch.from_numpy(im)
        # if torch.backends.mps.is_available:
        #     im.type(torch.MPSFloatType)
        im.to(self.seg_dev)
        bin_pred = None
        # now the importat part: the sliding window evaluation (or derivatives of it)
        pred_holder = None
        pred_lowres = None
        for model in self.Segment:
            pred = model(im)
            data_tpl['pred'] = pred

            # inside the postprocessing the result will be attached to the data_tpl
            if do_postprocessing:
                self.SegProcess.postprocess_data_tpl(data_tpl, 'pred', bin_pred)

            if type(pred_holder) == type(None):
                pred_holder = data_tpl['pred_orig_shape']
                pred_lowres = data_tpl['pred']
            else:
                pred_holder += data_tpl['pred_orig_shape']
                pred_lowres += data_tpl['pred']
                
        pred_holder = np.where(pred_holder>2,1,0)
        print("pred_holder max",pred_holder.max())
        return pred_holder, np.where(pred_lowres>2,1,0)
    
    
    def save_prediction(self, data_tpl, filename=None, key = 'pred_orig_shape',
                        save_npy=True):

        # find name of the file
        if filename is None:
            filename = data_tpl['scan'] + '.nii.gz'
        else:
            # remove fileextension e.g. .nii.gz
            filename = filename.split('.')[0] + '.nii.gz'



        key = 'pred_orig_shape'
        lr_key = 'pred_lowres'
        if not ('pred_orig_shape' in data_tpl):assert(1==2)
        if not ('pred_lowres' in data_tpl): assert(1==2)
            
        im_aff = self.save_nii_from_data_tpl(data_tpl, join(self.sv_fold_size, filename),key)
        if save_npy:
            self.save_npy_from_data_tpl(data_tpl, join(self.lrsv_fold_size, filename[:-7]),lr_key, aff = im_aff)
        
    def save_nii_from_data_tpl(self, data_tpl, out_file, key):
        arr = data_tpl[key]
    
        if not data_tpl['had_z_first']:
            arr = np.stack([arr[z] for z in range(arr.shape[0])], -1)
            
        if data_tpl['had_z_first']:
            for i in range(len(arr)):
                arr[i] = binary_fill_holes(arr[i])
        else:
            
            for i in range(len(arr[0,0])):
                arr[:,:,i] = binary_fill_holes(arr[:,:,i])  
                
        raw_path = join(environ['OV_DATA_BASE'], 'raw_data', data_tpl['dataset'])
        im_file = None
        if data_tpl['raw_image_file'].endswith('.nii.gz'):
            # if not the file was loaded from dcm
            if exists(data_tpl['raw_image_file']):
                im_file = data_tpl['raw_image_file']
            elif exists(raw_path):
                # ups! This happens when you've copied over the preprocessed data from one
                # system to antoher. We have to find the raw image file, but luckily everything
                # should be contained in the data_tpl to track the file
                im_folders = [imf for imf in listdir(raw_path) if imf.startswith('images')]
                im_file = []
                for imf in im_folders:
                    if basename(data_tpl['raw_image_file']) in listdir(join(raw_path, imf)):
                        im_file.append(join(raw_path, imf, basename(data_tpl['raw_image_file'])))
                        
    
        if im_file is not None:
            # if we have found a raw_image_file, we will use it to build the prediction nifti
            if isinstance(im_file, (list, tuple)):
                im_file = im_file[0]
            img = nib.load(im_file)
            nii_img = nib.Nifti1Image(arr, img.affine, img.header)
        else:
            # if we couldn't find anything (e.g. if the image was given as a DICOM)
            nii_img = nib.Nifti1Image(arr, np.eye(4))
            if key.endswith('orig_shape') and 'orig_spacing' in data_tpl:
                nii_img.header['pixdim'][1:4] = data_tpl['orig_spacing']
            else:
                nii_img.header['pixdim'][1:4] = data_tpl['spacing']

        nib.save(nii_img, out_file)
        return img.affine
        
    def save_npy_from_data_tpl(self, data_tpl, out_file, key, aff = None):
        arr = data_tpl[key]
    
        if not data_tpl['had_z_first']:
            arr = np.stack([arr[z] for z in range(arr.shape[0])], -1)

        if data_tpl['had_z_first']:
            for i in range(len(arr)):
                arr[i] = binary_fill_holes(arr[i])
        else:
            for i in range(len(arr[0,0])):
                arr[:,:,i] = binary_fill_holes(arr[:,:,i])  
                
        if not (aff is None):
            #ensures images always come in with a constant orientation, 
            #using their affine matrix from nifti file
            im = sitk.GetImageFromArray(arr)
            im.SetOrigin(-aff[:3,3])
            im.SetSpacing(data_tpl['orig_spacing'].astype(np.float16).tolist())
            ##flips image along correct axis according to image properties
            flip_im = sitk.Flip(im, np.diag(aff[:3,:3]<-0).tolist())
            arr = np.rot90(sitk.GetArrayViewFromImage(flip_im))
                
        np.save(out_file,arr)
 

    def Segment_CT(self, im_path:str =  None, save_path:str= None,
                   volume_thresholds = [250]):
        ##to convert the save process from .npy to .nii.gz we need to do the following:
        ## use from ovseg.utils.io import save_nii_from_data_tpl, which requires data to be in data_tpl form not volume form. This requires:
        ## we use Dataset dataloader to feed data to unet, as this generates the data_tpl for each scan. Effectively, we will be setting up Simstudy data as a validation
        #dataset. This requires that we provide: preprocessed data loc, scans, and 'keys' - I do not know what the keys are (in Dataset)
        self.Segment = []
        self._load_UNet(self.seg_mp_low,self.seg_dev)
        # self._load_UNet(self.seg_mp_high,self.seg_dev,res='high')
        
        self.preprocess_path = join(self.segpredpath,"preprocessed",self.data_name,self.preprocessed_name,"images")

        
        self.SegProcess = SegProcess(apply_small_component_removing= True,lb_classes=[1],
                                                  volume_thresholds = volume_thresholds,
                                                  remove_comps_by_volume=True,
                                                  use_fill_holes_3d = True)
        self.segmodel_parameters_low = np.load(join(self.seg_mp_low,"model_parameters.pkl"),allow_pickle=True)
        
        params_low = self.segmodel_parameters_low['data'].copy()
        params_low['folders'] = ['images']
        params_low['keys'] = ['image']

        self.segpp_data = SegmentationData(preprocessed_path=split(self.preprocess_path)[0],
                                     augmentation= None,
                                     **params_low)

        
        ppscans = [self.segpp_data.val_ds[i]['scan'] for i in range(len(self.segpp_data.val_ds))]
        
        
        
        for i in range(len(self.segpp_data.val_ds)):
            
            # get the data
            data_tpl = self.segpp_data.val_ds[i]
            filename = data_tpl['scan'] + '.nii.gz'
            print("Segmenting {}...".format(data_tpl['scan']))
            
            
            # first let's try to find the name
            if 'scan' in data_tpl.keys():
                scan = data_tpl['scan']
            else:
                d = str(int(np.ceil(np.log10(len(self.segpp_data)))))
                scan = 'case_%0'+d+'d'
                scan = scan % i
            # predict from this datapoint
            pred, pred_lowres = self.seg_pred(data_tpl)
                
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()
                
            
            data_tpl['pred_orig_shape'] = pred
            data_tpl['pred_lowres'] = pred_lowres
            self.save_prediction(data_tpl,filename=scan)
            print("")
    
        print("Segmentations complete!\n")
        

        
   
if __name__ == "__main__":
    seg_fp = "/home/wcm23/rds/hpc-work/CascadeSegmentationFiles/cascade_stage1/"
    # hpc_loc = '/bask/projects/p/phwq4930-renal-canc/data/ovseg_all_data/trained_models/coreg_ncct/4.0mm_allbinary_noR74K118'
    data_names = ['all_add']
    do_prep = True
    do_infer = True
        
    for data_name in data_names:
        test = Ensemble_Seg(data_name,
                              seg_fp = seg_fp,seg_dev='cuda',
                              spacing = np.array([4]*3),
                              do_prep=do_prep, do_infer = do_infer)
        
        torch.cuda.empty_cache()
        gc.collect()
    
            
        