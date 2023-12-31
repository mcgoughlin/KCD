import object_creation_utils as ocu
import pandas as pd
import os
import file_utils as fu
import shutil 

# COREG CREATION
dataset_CD = 'coreg_ncct'
home = '/Users/mcgoug01/Downloads/Data/'
im_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images/'.format(dataset_CD)
infnpy_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset_CD)
infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset_CD)
lb_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/labels/'.format(dataset_CD)
is_testing_code=False
overwrite=True
# ocu.create_labelled_dataset(home,dataset_CD,im_p,infnpy_p,infnii_p,lb_p,
#                         is_testing=is_testing_code,overwrite=overwrite)
save_dir_CD=os.path.join(home,'objects',dataset_CD)
fu.save_normalisation_params(save_dir_CD)

# KITS23 CREATION
dataset_k23 = 'kits23_nooverlap'
im_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images/'.format(dataset_k23)
infnpy_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset_k23)
infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset_k23)
lb_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/labels/'.format(dataset_k23)
is_testing_code=False
overwrite=False
# ocu.create_labelled_dataset(home,dataset_k23,im_p,infnpy_p,infnii_p,lb_p,
#                         is_testing=is_testing_code,overwrite=overwrite)
save_dir_k23=os.path.join(home,'objects',dataset_k23)
fu.save_normalisation_params(save_dir_k23)

# MERGING AND SAVING
df_CD = pd.read_csv(os.path.join(save_dir_CD,'features_labelled.csv'))
df_k23 = pd.read_csv(os.path.join(save_dir_k23,'features_labelled.csv'))
merged_features = pd.merge(df_CD,df_k23,how='outer',on=df_k23.columns.tolist())

new_dataset = 'merged_training_set'
save_dir_merged = os.path.join(home,'objects',new_dataset)
fu.create_folder(save_dir_merged)
folders = fu.setup_save_folders(save_dir_merged)
merged_features.to_csv(os.path.join(save_dir_merged,'features_labelled.csv'))
fu.save_normalisation_params(save_dir_merged)

for case in df_k23.case:
    for position in df_k23[df_k23.case == case].position:
        for directory in folders:
            _,folder = os.path.split(directory)
            if 'obj' in folder: ext = '.obj'
            else: ext = '.npy'
            filename = case[:-7]+'_'+position+ext
            src_fp = os.path.join(save_dir_k23,folder,filename)
            dest_fp = os.path.join(directory,filename)
            shutil.copy(src_fp,dest_fp)

for case in df_CD.case:
    for position in df_CD[df_CD.case == case].position:
        for directory in folders:
            _,folder = os.path.split(directory)
            if 'obj' in folder: ext = '.obj'
            else: ext = '.npy'
            filename = case[:-7]+'_'+position+ext
            src_fp = os.path.join(save_dir_CD,folder,filename)
            dest_fp = os.path.join(directory,filename)
            shutil.copy(src_fp,dest_fp)