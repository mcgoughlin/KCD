import object_creation_utils as ocu

dataset = 'add_ncct_unseen'
im_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images/'.format(dataset)
infnpy_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset)
infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset)
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}'.format(dataset)
is_testing_code=False
overwrite=False

ocu.create_unseen_dataset(dataset,im_p,infnpy_p,infnii_p,save_dir,
                      is_testing=is_testing_code,overwrite=overwrite)