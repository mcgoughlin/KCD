import os
os.environ['OV_DATA_BASE'] = '/home/wcm23/rds/hpc-work/FineTuningKITS23'

from KCD.Segmentation.Inference.EnsembleSeg import Ensemble_Seg
import numpy as np

seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/masked_coreg_ncct/2mm_binary/6,3x3x3,32_finetune_fromkits23no_detection'
# seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/all_cect/4mm_binary/6,3x3x3,32'
data_names = ['masked_test_set']
do_prep = True
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name,
                        seg_fp=seg_fp,
                        spacing=np.array([2] * 3),
                        do_prep=do_prep, do_infer=do_infer,
                        is_cect=False,cont=True)

