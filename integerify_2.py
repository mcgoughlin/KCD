import nibabel as nib
import os

# we want to load each .nii.gz label file from 'path', extract the flaot data, convert it to integer if value>0.5,
# and save the result as a new .nii.gz file in 'save_path'

path = '/Users/mcgoug01/Downloads/all_ncct/labels'
save_path = '/Users/mcgoug01/Downloads/all_ncct/integer_labels'

files = os.listdir(path)

for file in files:
    if file[-7:]!='.nii.gz':
        continue
    img = nib.load(path+'/'+file)
    data = img.get_fdata()
    data[data>0.5]=1
    data[data<=0.5]=0
    img = nib.Nifti1Image(data.astype(int), img.affine, img.header)
    nib.save(img, save_path+'/'+file)