# I want to copy files that exist in one directory to another directory,
# currently, the source files just contain the casenum '86.nii.gz',
# but I want the dest files to be preceeded by 'Rcc_', and always be 3-digit,
# filling the casenum with zeros if the number is less than 100, so 'Rcc_086.nii.gz'.

import os
import shutil

# Set the directories
source_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/test_set/add_ncct'
dest_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/test_set/images'
check_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/test_set/add_labels'
# Get the files
source_files = os.listdir(source_dir)
dest_files = os.listdir(dest_dir)
check_files = os.listdir(check_dir)

# Loop through the files
for file in source_files:
    # add trailing zeros to the file name
    desfile = file.split('.')[0]
    desfile = desfile.zfill(3)
    desfile = 'Rcc_{}.nii.gz'.format(desfile)

    if (desfile in dest_files) or (desfile not in check_files):
        print('File {} already exists in {}'.format(desfile,dest_dir))
    else:
        shutil.copy(os.path.join(source_dir,file),os.path.join(dest_dir,desfile))
        print('Copied {} to {}'.format(desfile,dest_dir))