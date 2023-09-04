import object_creation_utils as ocu

dataset = 'coreg_ncct'
im_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images/'.format(dataset)
infnpy_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset)
infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset)
lb_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/labels/'
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}'.format(dataset)
is_testing_code=False
overwrite=False


ocu.create_labelled_dataset(dataset,im_p,infnpy_p,infnii_p,lb_p,save_dir,
                        is_testing=is_testing_code,overwrite=overwrite)

dataset = 'kits23sncct'
im_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images/'.format(dataset)
infnpy_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset)
infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/'.format(dataset)
lb_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/labels/'.format(dataset)
save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}'.format(dataset)
is_testing_code=False
overwrite=False


ocu.create_labelled_dataset(dataset,im_p,infnpy_p,infnii_p,lb_p,save_dir,
                        is_testing=is_testing_code,overwrite=overwrite)