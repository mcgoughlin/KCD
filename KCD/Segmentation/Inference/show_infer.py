#this script is used to display the segmentation inference overlayed on the original ct scan

import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

path = '/Users/mcgoug01/Downloads/test_data'
cancer_infp = os.path.join(path, 'cancer_inferences')
masked_ct_path = os.path.join(path, 'images')
ct_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/add_ncct/images'
kidney_infp = os.path.join(path, 'kidney_inferences')

#take the first file in the directory
file = [path for path in os.listdir(cancer_infp) if path.endswith('.nii.gz')][-5]
cancer_inf = nib.load(os.path.join(cancer_infp, file))
masked_ct = nib.load(os.path.join(masked_ct_path, file))
ct = nib.load(os.path.join(ct_path, file))
kidney_inf = nib.load(os.path.join(kidney_infp, file))


# plot the ct scan on the slice with the highest total cancerous inference value
cancer_inf = cancer_inf.get_fdata()
masked_ct = masked_ct.get_fdata()
ct = ct.get_fdata()
kid = kidney_inf.get_fdata()
print(cancer_inf.shape, masked_ct.shape, ct.shape, kid.shape)

slice = np.argmax(np.sum(cancer_inf, axis=(0,1)))

masked_ct_slice =  np.rot90(masked_ct[:,:,slice],3)
cancer_inf_slice = np.rot90(cancer_inf[:,:,slice],3)
ct_slice = np.rot90(ct[:,:,-slice],3)
kid_slice = np.rot90(kid[:,:,-slice],3)

plt.switch_backend('TkAgg')
plt.figure()
plt.subplot(1,3,1)
plt.imshow(ct_slice, cmap='gray',vmin=-200, vmax=200)
plt.title('Original NCCT Scan')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(ct_slice, cmap='gray',vmin=-200, vmax=200)
plt.imshow(kid_slice, alpha=kid_slice)
plt.title('Stage 1: Kidney Segmentation')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(masked_ct_slice, cmap='gray',vmin=-200, vmax=200)
plt.imshow(cancer_inf_slice, alpha=cancer_inf_slice/1000, cmap='Reds')
plt.title('Stage 2: Cancer Segmentation')
plt.axis('off')
plt.show()