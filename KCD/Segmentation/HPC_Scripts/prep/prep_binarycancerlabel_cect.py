import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data/"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

data_name = 'masked_kits23_nooverlap'
spacing = 1

preprocessed_name = '1mm_binary_canceronly'

lb_classes = [2]
target_spacing=[spacing]*3

prep = SegmentationPreprocessing(apply_resizing=True, 
                                    apply_pooling=False, 
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    reduce_lb_to_single_class = True)

prep.plan_preprocessing_raw_data(raw_data=data_name)

prep.preprocess_raw_data(raw_data=data_name,
                          preprocessed_name=preprocessed_name)