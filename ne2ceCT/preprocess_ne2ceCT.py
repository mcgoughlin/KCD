import os
os.environ['OV_DATA_BASE'] = '/media/mcgoug01/Crucial X6/ovseg_test/'
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import numpy as np


def Segmentation_Preparation(home, seg_spacing=np.array([4,4,4]),
                             data_name='test',preprocessed_name='test', is_cect=False):

    pp_save_path = os.path.join(home, "preprocessed", data_name,preprocessed_name, 'images')

    rawdata_path = os.path.join(home, "raw_data", data_name, 'images')

    print("##SEG PREPROCESS##\nPreprocessing CT Volumes to {}\n Stored in location {}.".format(seg_spacing,
                                                                                               pp_save_path))
    print("")
    if is_cect:
        preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                                  apply_pooling=False,
                                                  apply_windowing=True,
                                                  target_spacing=seg_spacing,
                                                  pooling_stride=None,
                                                  window=np.array([-556.0, 309.2]),
                                                  scaling=np.array([199.5, 71.6]),
                                                  lb_classes=[1,2],
                                                  reduce_lb_to_single_class=False,
                                                  lb_min_vol=None,
                                                  prev_stages=[],
                                                  save_only_fg_scans=False,
                                                  n_im_channels=1)
    else:
        preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                                  apply_pooling=False,
                                                  apply_windowing=True,
                                                  target_spacing=seg_spacing,
                                                  pooling_stride=None,
                                                  window=np.array([-116., 130.]),
                                                  scaling=np.array([41.301857, 12.257426]),
                                                  lb_classes=[1,2],
                                                  reduce_lb_to_single_class=False,
                                                  lb_min_vol=None,
                                                  prev_stages=[],
                                                  save_only_fg_scans=False,
                                                  n_im_channels=1)

    preprocessing.preprocess_raw_data(raw_data=data_name,
                                      preprocessed_name=preprocessed_name,
                                      data_name=None,
                                      save_as_fp16=True)
    print("")

if __name__ == "__main__":
    Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
                             data_name='small_coreg_ncct',preprocessed_name='small_coreg_ncct_2', is_cect=False)

    Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
                             data_name='large_coreg_ncct',preprocessed_name='large_coreg_ncct_2', is_cect=False)

    # Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='add_cect',preprocessed_name='add_cect_2', is_cect=False)
