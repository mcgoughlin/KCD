import nibabel as nib
import numpy as np
import os
import torch
from scipy.ndimage.morphology import binary_fill_holes
import scipy.ndimage as spim
from skimage.measure import regionprops
from KCD.Detection.Preprocessing.ObjectFiles import object_creation_utils as ocu
from skimage.segmentation import watershed

# we wan to find the optimum confidence threshold for determining the presence of cancerous voxel labels,
# and the optimum size threshold for determining the presence of a cancerous region
# we will do this by finding the threshold that maximises the dice score on a dataset

path = '/Users/mcgoug01/Downloads/test_data'
cancer_infp = os.path.join(path, 'cancer_test_inferences')
kidney_infp = os.path.join(path, 'kid_test_inferences')
confidence_thresholds = np.arange(0, 1, 0.02)
vol = 400

results =[]

#index 0 is left kidney, index 1 is right kidney
test_labels = {'20.nii.gz':[0,0],
                '74.nii.gz':[0,0],
                '131.nii.gz':[0,0],
                '132.nii.gz':[0,0],
                '133.nii.gz':[0,0],
                '135.nii.gz':[0,0],
                '136.nii.gz':[0,0],
                '137.nii.gz':[0,0],
                '138.nii.gz':[0,0],
                '139.nii.gz':[0,0],
                '189.nii.gz':[0,0],
                '191.nii.gz':[0,0],
                '396.nii.gz':[0,0],
                '397.nii.gz':[0,0],
                '398.nii.gz':[0,0],
                '767.nii.gz':[0,0],
                '772.nii.gz':[0,0],
                'Rcc_002.nii.gz':[1,0],
                'Rcc_005.nii.gz':[1,0],
                'Rcc_009.nii.gz':[0,1],
                'Rcc_010.nii.gz':[1,0],
                'Rcc_012.nii.gz':[1,0],
                'Rcc_018.nii.gz':[0,1],
                'Rcc_021.nii.gz':[0,1],
                'Rcc_022.nii.gz':[0,1],
                'Rcc_024.nii.gz':[0,1],
                'Rcc_026.nii.gz':[0,1],
                'Rcc_029.nii.gz':[1,0],
                'Rcc_036.nii.gz':[0,1],
                'Rcc_048.nii.gz':[1,0],
                'Rcc_056.nii.gz':[1,0],
                'Rcc_063.nii.gz':[1,0],
                'Rcc_065.nii.gz':[1,0],
                'Rcc_070.nii.gz':[1,0],
                'Rcc_073.nii.gz':[1,0],
                'Rcc_077.nii.gz':[0,1],
                'Rcc_079.nii.gz':[0,1],
                'Rcc_080.nii.gz':[1,0],
                'Rcc_086.nii.gz':[0,1],
                'Rcc_091.nii.gz':[1,0],
                'Rcc_092.nii.gz':[1,0],
                'Rcc_094.nii.gz':[0,1],
                'Rcc_097.nii.gz':[1,0],
                'Rcc_098.nii.gz':[0,1],
                'Rcc_105.nii.gz':[1,0],
                'Rcc_106.nii.gz':[0,1],
                'Rcc_109.nii.gz':[1,0],
                'Rcc_110.nii.gz':[1,0],
                'Rcc_112.nii.gz':[1,0],
                'Rcc_119.nii.gz':[1,0],
                'Rcc_130.nii.gz':[0,1],
                'Rcc_133.nii.gz':[1,0],
                'Rcc_135.nii.gz':[0,1],
                'Rcc_137.nii.gz':[0,1],
                'Rcc_139.nii.gz':[0,1],
                'Rcc_159.nii.gz':[1,0],
                'Rcc_163.nii.gz':[1,0],
                'Rcc_165.nii.gz':[1,0],
                'Rcc_169.nii.gz':[0,0],
                'Rcc_175.nii.gz':[0,1],
                'Rcc_184.nii.gz':[0,1],
                'Rcc_187.nii.gz':[0,1],
                'Rcc_191.nii.gz':[1,0],
                'Rcc_196.nii.gz':[1,0],
                'Rcc_202.nii.gz':[0,1]}

#only load files that end in .nii.gz
for conf in confidence_thresholds:
    tp, fp, fn, tn = 0, 0, 0, 0
    for file in [path for path in os.listdir(cancer_infp) if path.endswith('.nii.gz')]:
        cancer_inf = nib.load(os.path.join(cancer_infp, file))
        kidney_inf = nib.load(os.path.join(kidney_infp, file))

        left_label,right_label = test_labels[file]
        # #find spacing
        spacing = cancer_inf.header['pixdim'][1:4]

        #resize all to 2mm spacing using torch
        cancer_inf = torch.from_numpy(cancer_inf.get_fdata()).unsqueeze(0).unsqueeze(0)
        kidney_inf = torch.from_numpy(kidney_inf.get_fdata()).unsqueeze(0).unsqueeze(0)
        cancer_inf = torch.nn.functional.interpolate(cancer_inf, scale_factor=tuple(spacing/2), mode='trilinear')
        kidney_inf = torch.nn.functional.interpolate(kidney_inf, scale_factor=tuple(spacing/2), mode='trilinear')

        #find cancerous voxels
        cancer_voxels = cancer_inf > conf
        # fill holes in cancerous voxels
        cancer_voxels = binary_fill_holes(cancer_voxels)
        #filter out regions with less than vol voxels using regionprops and spim
        cancer_voxels = spim.label(cancer_voxels.squeeze())[0]
        cancer_regions = np.array([region for region in regionprops(cancer_voxels) if region.area > vol])

        # algorithm:
        # generate cancer_voxels, colour in kidney labels in 2 if cancerous region overlaps, 1 if not
        # any cancer region that does not overlap with a kidney label will be its own cancerous region in cancer_voxels
        # then, refer to test labels to count results.

        # any regions highlighted as kidney that arent actually kidney, safely ignore.
        # any regions highlighted as cancer that arent actually cancer, this is a false positive.

        #first, check if kidney inference contains only 2 regions - if so, can easily apply left and right labels
        # if not, then check if the left and right labels are the same - if so, apply the same label to all regions
        # if not, then we have a problem and need to manually check the inference

        #TLDR: start with kidney inferences - if 2, then simply apply left and right labels, and use cancer_voxels
        # to check answers. if not 2, then check if left and right labels are the same - if so, apply same label to all
        # regions, and use cancer_voxels to check answers. if not, then manually check inference.


    entry = {'confidence_threshold': conf, 'size_threshold': vol, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    sensitivity = tp/(tp+fn+1e-6)
    specificity = tn/(tn+fp+1e-6)
    precision = tp/(tp+fp+1e-6)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    entry['sensitivity'] = sensitivity
    entry['specificity'] = specificity
    entry['precision'] = precision
    entry['accuracy'] = accuracy
    entry['dice'] = 2*tp/(2*tp+fp+fn+1e-6)

    print(entry)
    results.append(entry)

import pandas
df = pandas.DataFrame(results)
df.to_csv(os.path.join(path, 'results_400_validation.csv'))