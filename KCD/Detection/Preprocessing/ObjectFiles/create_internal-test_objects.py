import object_creation_utils as ocu
import file_utils as fu
import os
dataset = 'add_ncct_unseen'
home = '/Users/mcgoug01/Downloads/Data'
im_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images/'.format(dataset)
infnpy_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset)
infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset)
is_testing_code=False
overwrite=False
# ocu.create_unseen_dataset(home,dataset,im_p,infnpy_p,infnii_p,is_testing=is_testing_code,overwrite=overwrite)
fu.save_normalisation_params(os.path.join(home,'objects',dataset),is_labelled=False)
