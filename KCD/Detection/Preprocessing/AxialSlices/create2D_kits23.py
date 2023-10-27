import slice_generating_utils as generator
import numpy as np

data_name = 'kits23'
path = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images".format(data_name)
save_dir = '/Users/mcgoug01/Downloads/Data'
segpath = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/labels".format(data_name)

overlap_mm = 40 ## this dictates the minimum distance apart between each slice! not the overlap.
patch2d = 224 ## in plane slice spacing
save_limit_percase_perlabel = 50 ## maximum number of saved shifted window slices per kidney
voxel_spacings = [1] ## isotropic voxel size in mm
thresholds_r_mm = [10] ## threshold used to calculate cancer label
kidney_r_mm = 20 ## threshold used to calculate kidney label
depth_z = 1 ## size of slice depth in axial dimension in mm - if 1mm, then 2D
bbox_boundary_mm = 40 ## dilation of segmentation label
boundary_z= 1 ## axial spacing between slice sampling
has_seg_label = True ## if we don't have seg label, we only generated foreground/background slices

patch_size = np.array([patch2d]*3) 
patch_size[0]=depth_z

for thresh in thresholds_r_mm:
    for spacing in voxel_spacings: 
        target_spacing = np.array([spacing]*3)
        overlap = overlap_mm/(patch_size*spacing)
        # overlap_mm = spacing*patch2d/10
        print("Voxel Spacing {}mm".format(spacing))
        generator.create_labelled_dataset(path,save_dir,segpath,target_spacing,overlap,
                        patch_size,thresh,spacing,save_limit_percase_perlabel,
                        bbox_boundary_mm,data_name=data_name,boundary_z=boundary_z,
                        depth_z=depth_z,kidney_thresh_rmm=kidney_r_mm)
        
for thresh in thresholds_r_mm:
    for spacing in voxel_spacings: 
        target_spacing = np.array([spacing]*3)
        overlap = overlap_mm/(patch_size*spacing)
        # overlap_mm = spacing*patch2d/10
        print("Voxel Spacing {}mm".format(spacing))
        generator.create_labelled_dataset(path,save_dir,segpath,target_spacing,overlap,
                       patch_size,0,spacing,save_limit_percase_perlabel,
                       bbox_boundary_mm,data_name=data_name,boundary_z=boundary_z,
                       depth_z=depth_z,kidney_thresh_rmm=0)