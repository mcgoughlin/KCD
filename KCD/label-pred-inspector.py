import nibabel as nib
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def get_nifti_data(path):
    img = nib.load(path)
    return img.get_fdata()

def get_numpy_data(path):
    return np.load(path,allow_pickle=True)

npy_impath = '/Users/mcgoug01/Downloads/all_cect_npy/images/'
npy_labpath = '/Users/mcgoug01/Downloads/all_cect_npy/labels/'
nii_labpath = '/Users/mcgoug01/Downloads/all_cect/labels/'
nii_impath = '/Users/mcgoug01/Downloads/all_cect/images/'
npy_files = os.listdir(npy_labpath)

for file in npy_files:
    if file[-4:]!='.npy':
        continue
    if not os.path.exists(nii_labpath+file[:-4]+'.nii.gz'):
        continue

    im_pp = get_numpy_data(npy_impath+file)
    lbl = get_numpy_data(npy_labpath+file)
    print(lbl.shape)

    im = get_nifti_data(nii_impath+file[:-4]+'.nii.gz')
    pred = get_nifti_data(nii_labpath+file[:-4]+'.nii.gz')
    print(pred.shape)

    fig = plt.figure(figsize=(10,5))
    fig.suptitle(file[:-4])

    plt.subplot(1,2,2)
    index = lbl.sum(axis=(2,3)).argmax()
    print(index, lbl.max(), pred.max())
    if file.startswith('KiTS'):
        pred_index = int((pred.shape[0]/lbl.shape[1])*index)
        plt.imshow(im[pred_index], alpha=0.5,vmax=200,vmin=-200)
        plt.imshow(pred[pred_index], alpha=0.5)
    else:
        pred_index = int((pred.shape[2]/lbl.shape[1])*index)
        plt.imshow(im[:,:,pred_index],alpha=0.5,vmax=200,vmin=-200)
        plt.imshow(pred[:,:,pred_index],alpha=0.5)


    plt.title('pred')
    plt.subplot(1,2,1)
    plt.title('label')
    plt.imshow(im_pp[0, index, :, :],alpha=0.5)
    plt.imshow(lbl[0,index,:,:],alpha=0.5)
    plt.show(block=True)
