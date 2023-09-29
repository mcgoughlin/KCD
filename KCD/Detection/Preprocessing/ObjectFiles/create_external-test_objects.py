import object_creation_utils as ocu
import file_utils as fu
import os
dataset = 'CTORG_all'
home = '/bask/projects/p/phwq4930-renal-canc/KCD_data/Data'
im_p = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/{}/images/'.format(dataset)
infnpy_p = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset)
infnii_p = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset)
is_testing_code=False
overwrite=False
ocu.create_unseen_dataset(home,dataset,im_p,infnpy_p,infnii_p,is_testing=is_testing_code,overwrite=overwrite)
fu.save_normalisation_params(os.path.join(home,'objects',dataset),is_labelled=False)
