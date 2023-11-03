import os
os.environ['OV_DATA_BASE'] = '/home/wcm23/rds/hpc-work/FineTuningKITS23'

from KCD.Segmentation.Inference.EnsembleSeg import Ensemble_Seg
import numpy as np

# seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/coreg_ncct/4.0mm_alllabels/cect2coreg_finetune'
seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/kits23/4.0mm_binary/6,3x3x3,32'
data_names = ['kits23']
do_prep = True
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name,
                        seg_fp=seg_fp,
                        spacing=np.array([4] * 3),
                        do_prep=do_prep, do_infer=do_infer,is_cect=True)