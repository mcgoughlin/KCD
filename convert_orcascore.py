import SimpleITK as sitk
import os

path = '/Users/mcgoug01/Downloads/orcascore/Challenge data'
subfolders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
sv_folds = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder+'nii'))]
print(subfolders)
for sv_fold,subfolder in zip(subfolders,sv_folds):
    if not os.path.exists(sv_fold):
        os.makedirs(sv_fold)
    images = [image for image in os.listdir(os.path.join(path, subfolder,'Images')) if image.endswith('.mhd')]
    for image in images:
        img = sitk.ReadImage(os.path.join(path, subfolder,'Images', image))
        sitk.WriteImage(img, os.path.join(sv_fold, image.replace('.mhd', '.nii.gz')))

# create 'ncct' and 'cect' folders - if file ends in 'AI.nii.gz' move to 'cect' else 'ncct'
phase_folders = ['ncct', 'cect']
for phase in phase_folders:
    if not os.path.exists(os.path.join(path, phase)):
        os.makedirs(os.path.join(path, phase))
for phase in phase_folders:
    for subfolder in subfolders:
        images = [image for image in os.listdir(os.path.join(path, subfolder)) if image.endswith('.nii.gz')]
        for image in images:
            if image.endswith('AI.nii.gz'):
                os.rename(os.path.join(path, subfolder, image), os.path.join(path, 'cect', image))
            else:
                os.rename(os.path.join(path, subfolder, image), os.path.join(path, 'ncct', image))
