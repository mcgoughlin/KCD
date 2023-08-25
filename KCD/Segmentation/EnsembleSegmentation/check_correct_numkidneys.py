

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.measure import regionprops
import scipy.ndimage as spim

im_p = '/media/mcgoug01/nvme/SecondYear/Classification/hq_seginferences/images/'
lb_p = '/media/mcgoug01/nvme/SecondYear/Classification/hq_seginferences/labels/'

cases = os.listdir(im_p)

def nifti_2_correctarr(im_n):
    aff = im_n.affine
    im = sitk.GetImageFromArray(im_n.get_fdata())
    im.SetOrigin(-aff[:3,3])
    im.SetSpacing(im_n.header['pixdim'][1:4].tolist())
    
    ##flips image along correct axis according to image properties
    flip_im = sitk.Flip(im, np.diag(aff[:3,:3]<-0).tolist())
    
    
    nda = np.rot90(sitk.GetArrayViewFromImage(flip_im))
    return nda.copy()

for case in cases:
    im_n = nib.load(os.path.join(im_p,case))
    lb_n = nib.load(os.path.join(lb_p,case))

    im = nifti_2_correctarr(im_n)
    seg = nifti_2_correctarr(lb_n)
    
    kidneys = [mass for mass in regionprops(spim.label(seg)[0])]
    
    if len(kidneys)!=2:
        print(case, "has {} kidneys.".format(len(kidneys)))
    
    
    # nda[100,:] = np.zeros(nda[100,:].shape)
    # extent = nda.shape[-1]
    # plt.imshow(nda[:,:,int(extent/2)])
    # plt.scatter([100],[250])
    # plt.show(block=True)
    
    