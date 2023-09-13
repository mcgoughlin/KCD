from KCD.Segmentation.Inference.EnsembleSeg import Ensemble_Seg
import numpy as np

# seg_fp = "/home/wcm23/rds/hpc-work/CascadeSegmentationFiles/cascade_stage1/"
home = 'basker/projects/p/phwq4930-renal-canc/data/seg_data'
seg_fp = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/coreg_ncct/4.0mm_allbinary_noR74K118/6,3x3x3,32_pretrained_noerrors'
data_names = ['coreg_ncct']
do_prep = True
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(home, data_name,
                        seg_fp=seg_fp, seg_dev='cuda',
                        spacing=np.array([4] * 3),
                        do_prep=do_prep, do_infer=do_infer)