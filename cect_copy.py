# this script will copy files of naming convention case_XXXXX.nii.gz to a new directory
# and rename them to KiTS-XXXXX.nii.gz

import os
import shutil
import glob

src_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/kits23/2.0mm_binary/images'
dest_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/coltea_add_kits_cect/coltea_add_cect_2/images'

src_files = [file for file in os.listdir(src_dir) if file.startswith('case_') and file.endswith('.npy')]
print(src_files)
for file in src_files:
    new_file = file.replace('case_', 'KiTS-')
    print(new_file)
    shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, new_file))