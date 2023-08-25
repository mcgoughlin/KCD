

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

path1 = '/media/mcgoug01/nvme/SecondYear/Segmentation/ensemble_seg/raw_data/kits_ncct/images/'
path2 = '/media/mcgoug01/nvme/SecondYear/Segmentation/ensemble_seg/raw_data/add_ncct/images/'
case1 = 'KiTS-00000.nii.gz'
case2 = 'RCC_012.nii.gz'

fp = os.path.join(path2,case2)
im_n = nib.load(fp)
aff = im_n.affine
im = sitk.GetImageFromArray(im_n.get_fdata())
im.SetOrigin(-aff[:3,3])
im.SetSpacing(im_n.header['pixdim'][1:4].tolist())

##flips image along correct axis according to image properties
flip_im = sitk.Flip(im, np.diag(aff[:3,:3]<-0).tolist())


nda = np.rot90(sitk.GetArrayViewFromImage(flip_im))
nda = nda.copy()
nda[100,:] = np.zeros(nda[100,:].shape)
extent = nda.shape[-1]
plt.imshow(nda[:,:,int(extent/2)])
plt.scatter([100],[250])