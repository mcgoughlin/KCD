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


npy_label_loc = str(sys.argv[1]) # '/Users/mcgoug01/Downloads/inferences/[4 4 4]mm'
nii_image_loc = str(sys.argv[2]) # '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/images'
save_loc = str(sys.argv[3]) # '/Users/mcgoug01/Downloads/masked_coreg_ncct/images'
npy_label_list = os.listdir(npy_label_loc)
nii_image_list = os.listdir(nii_image_loc)

if not os.path.exists(save_loc):
    os.makedirs(save_loc)

#we want to dilate label 40x40x40mm, so 10x10x10 voxels

for npy_label in npy_label_list:
    if '.DS_Store' in npy_label:
        continue
    image = nib.load(os.path.join(nii_image_loc, npy_label.replace('npy', 'nii.gz')))
    label = np.load(os.path.join(npy_label_loc, npy_label))
    dilated_label = binary_dilation(label, iterations=10)
    image_data = image.get_fdata()
    # image_data = np.rot90(image_data, axes=(0, 1), k=3)
    # image_data = np.flip(np.flip(image_data, axis=-1), 1)

    if 'RCC' in npy_label:
        # interpolate label spacing from (4x4x4) to image spacing in the 'spacing' variable
        label_expanded = \
        nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(dilated_label), dim=0), dim=0),
                                  mode='nearest', size=image_data.shape).numpy()[0, 0]
        print(label_expanded.shape, image_data.shape)
        masked_image = np.where(label_expanded == 0, -500, image_data)
    else:
        # reshape so this is first axis - find the uncommon axis and move it to the front
        mode, counts = stats.mode(np.array(dilated_label.shape), axis=0)
        shifted_label = np.moveaxis(dilated_label, np.argmax(np.array(dilated_label.shape) != mode),
                                    np.argmax(np.array(image_data.shape) != 1))
        # interpolate label spacing from (4x4x4) to image spacing in the 'spacing' variable
        label_expanded = \
        nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(shifted_label), dim=0), dim=0),
                                  mode='nearest', size=image_data.shape).numpy()[0, 0]

        masked_image = np.where(label_expanded == 0, -500, image_data)


    # create masked nifti image and save
    masked_nifti = nib.Nifti1Image(masked_image, affine=image.affine)
    nib.save(masked_nifti, os.path.join(save_loc, npy_label.replace('npy', 'nii.gz')))

