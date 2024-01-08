# quick script that loads an .nii.gz image and its corresponding seg label, dilates the label,
# and then masks the image with the dilated label.

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
import torch
import torch.nn as nn
from scipy import stats
import sys
from KCD.Detection.Preprocessing.AxialSlices import array_manipulation_utils as amu

nii_label_loc = str(sys.argv[1]) # '/Users/mcgoug01/Downloads/inferences/[4 4 4]mm'
nii_image_loc = str(sys.argv[2]) # '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/images'
save_loc =  str(sys.argv[3]) # '/Users/mcgoug01/Downloads/masked_coreg_ncct/images'
npy_label_list = os.listdir(nii_label_loc)
nii_image_list = os.listdir(nii_image_loc)

if not os.path.exists(save_loc):
    os.makedirs(save_loc)

#we want to dilate label 40x40x40mm, so 10x10x10 voxels
for nii_label in npy_label_list:
    print(nii_label)
    if '.DS_Store' in nii_label:
        continue
    image = nib.load(os.path.join(nii_image_loc, nii_label))
    label = nib.load(os.path.join(nii_label_loc, nii_label))
    label_data = amu.nifti_2_correctarr(label)
    label_spacing = np.abs(image.header['pixdim'][1:4])
    #find mode of label shape
    shape_mode, _ = stats.mode(np.array(label_data.shape), axis=0)
    spacing_mode, _ = stats.mode(np.array(label_spacing), axis=0)
    non_modal_spacing = label_spacing[np.array(label_spacing) != spacing_mode]
    non_modal_shape = np.array(label_data.shape)[np.array(label_data.shape) != shape_mode]

    #compute scale_factor to resize label to 4x4x4mm voxels based on the shape of the label
    scale_factors = np.array([spacing_mode / 4 if shape == shape_mode else non_modal_spacing[0] / 4 for shape in label_data.shape])

    #resize voxels to 4x4x4mm and dilate label by 10x10x10 voxels
    label_4 = \
    nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(label_data), dim=0), dim=0),
                              mode='nearest', scale_factor = tuple(scale_factors)).numpy()[0, 0]
    dilated_label = binary_dilation(label_4, iterations=10)

    #return label to original shape
    dilated_label = \
    nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(dilated_label), dim=0), dim=0),
                                mode='nearest', size=label_data.shape).numpy()[0, 0]

    image_data = amu.nifti_2_correctarr(image)
    masked_image = np.where(dilated_label == 0, -500, image_data)
    spacing_vector = scale_factors * np.array([4,4,4])

    # rearrange so that the non-modal axes are at the end of the array - use modal and non-modal of label calculated earlier
    while masked_image.shape[-1] == shape_mode:
        if masked_image.shape[0] != shape_mode:
            masked_image = np.transpose(masked_image, (2,1,0))
            spacing_vector = spacing_vector[[2,1,0]]
        elif masked_image.shape[1] != shape_mode:
            masked_image = np.transpose(masked_image, (0,2,1))
            spacing_vector = spacing_vector[[0,2,1]]

    #generate new affine matrix
    affine = np.eye(4)*np.append(spacing_vector,1)

    # create masked nifti image and save
    masked_nifti = nib.Nifti1Image(np.rot90(masked_image,3), affine=affine)
    nib.save(masked_nifti, os.path.join(save_loc, nii_label))
