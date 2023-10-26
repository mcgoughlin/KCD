import os
os.environ['OV_DATA_BASE'] = '/bask/projects/p/phwq4930-renal-canc/data/seg_data'

from KCD.Segmentation.Inference.EnsembleSeg import Ensemble_Seg
import numpy as np

# seg_fp = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/coreg_ncct/4.0mm_allbinary_noR74K118/6,3x3x3,32_pretrained_noerrors'
seg_fp = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/all_cect/4mm_binary/6,3x3x3,32'
data_names = ['kits23']
do_prep = False
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name,
                        seg_fp=seg_fp,
                        spacing=np.array([4] * 3),
                        do_prep=do_prep, do_infer=do_infer)