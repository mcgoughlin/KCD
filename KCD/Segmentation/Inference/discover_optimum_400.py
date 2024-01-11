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
cancer_infp = os.path.join(path, 'cancer_inferences')
cancer_gt = os.path.join(path, 'cancer_labels')
kidney_infp = os.path.join(path, 'kid_inferences')

confidence_thresholds = np.arange(0, 1, 0.02)
vol = 400

results =[]

#only load files that end in .nii.gz
for conf in confidence_thresholds:
    tp, fp, fn, tn = 0, 0, 0, 0
    for file in [path for path in os.listdir(cancer_infp) if path.endswith('.nii.gz')]:
        cancer_inf = nib.load(os.path.join(cancer_infp, file))
        cancer_label = nib.load(os.path.join(cancer_gt, file))
        kidney_inf = nib.load(os.path.join(kidney_infp, file))
        # assert all images have same shape
        assert cancer_inf.shape == cancer_label.shape == kidney_inf.shape
        #find spacing
        spacing = cancer_inf.header['pixdim'][1:4]

        #resize all to 2mm spacing using torch
        cancer_inf = torch.from_numpy(cancer_inf.get_fdata()).unsqueeze(0).unsqueeze(0)
        cancer_label = torch.from_numpy(cancer_label.get_fdata()).unsqueeze(0).unsqueeze(0)
        kidney_inf = torch.from_numpy(kidney_inf.get_fdata()).unsqueeze(0).unsqueeze(0)
        cancer_inf = torch.nn.functional.interpolate(cancer_inf, scale_factor=tuple(spacing/2), mode='trilinear')
        cancer_label = torch.nn.functional.interpolate(cancer_label, scale_factor=tuple(spacing/2), mode='trilinear', align_corners=False)
        kidney_inf = torch.nn.functional.interpolate(kidney_inf, scale_factor=tuple(spacing/2), mode='trilinear', align_corners=False)



        #find cancerous voxels
        cancer_voxels = cancer_inf > conf
        # fill holes in cancerous voxels
        cancer_voxels = binary_fill_holes(cancer_voxels)
        #filter out regions with less than vol voxels using regionprops and spim
        cancer_voxels = spim.label(cancer_voxels.squeeze())[0]
        cancer_regions = np.array([region for region in regionprops(cancer_voxels) if region.area > vol])

        # reconstruct cancerous voxels from regions over vol
        cancer_voxels = np.zeros_like(cancer_voxels)
        for region in cancer_regions:
            cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]] = 2

        # colour kidney labels in 2 if cancerous, 1 if not
        travel_mask1 = cancer_label.squeeze().numpy().copy()
        #set 0 values to -1, set 1 values to 0 in travel mask
        travel_mask1[travel_mask1==0] = -1
        travel_mask1[travel_mask1==1] = 0
        cancer_kidney_labels = watershed(travel_mask1, markers=(cancer_label.numpy().squeeze()==2).astype(int),
                                  mask = (cancer_label.numpy().squeeze()>0).astype(int))
        cancer_kidney_labels[cancer_kidney_labels==1] = 2
        cancer_kidney_labels[(cancer_kidney_labels==0) & (cancer_label.numpy().squeeze()>0)] = 1

        # reconstruct labels voxels from cancer_kidney_labels, removing tiny regions
        label_voxels = np.zeros_like(cancer_kidney_labels)
        for label in [1,2]:
            for region in regionprops(spim.label(cancer_kidney_labels==label)[0]):
                if region.area > 20:
                    label_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]] = label


        # loop through all cancer predicted regions without a corresponding ground truth label - if they
        # predict cancer, its a false positive
        for region in regionprops(spim.label(cancer_voxels==2)[0]):
            label_region = label_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]]
            if label_region.max() < 1:
                fp+=1
        # loop through all cancerous labels - checking if true or false positive
        for region in regionprops(spim.label(label_voxels==2)[0]):
            prediction_region = cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]]
            if prediction_region.max() <2:
                fn+=1
            else:
                tp+=1
        # loop through all healthy labels - checking if true or false negative
        for region in regionprops(spim.label(label_voxels==1)[0]):
            prediction_region = cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]]
            if prediction_region.max() <2:
                tn+=1
            else:
                fp+=1

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