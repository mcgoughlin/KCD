#convert all Rcc_XXX.npy files to RCC_XXX.npy files

import shutil
import os

path = '/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/add_cect/add_cect_2/images/'
temp_loc = '/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/add_cect/add_cect_2/'
files = os.listdir(path)
for file in files:
    if 'Rcc' in file:
        # move the file and move it back with the new name
        shutil.move(os.path.join(path,file), os.path.join(temp_loc,file))
        shutil.move(os.path.join(temp_loc,file), os.path.join(path, file.replace('Rcc','RCC')))
        print('renamed',file)
    else:
        print('not renamed',file)

path = '/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/add_cect/add_cect_2/labels/'
files = os.listdir(path)
for file in files:
    if 'Rcc' in file:
        shutil.move(os.path.join(path,file), os.path.join(temp_loc,file))
        shutil.move(os.path.join(temp_loc,file), os.path.join(path, file.replace('Rcc','RCC')))
        print('renamed',file)
    else:
        print('not renamed',file)