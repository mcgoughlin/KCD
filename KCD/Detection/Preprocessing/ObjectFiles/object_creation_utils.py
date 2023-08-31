import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.spatial import distance
from skimage.measure import regionprops,marching_cubes
import scipy.ndimage as spim
from scipy import stats
import test_utils as tu
import feature_extraction_utils as feu
import graph_smoothing_utils as gmu
import file_utils as fu

def nifti_2_correctarr(im_n):
    aff = im_n.affine
    im = sitk.GetImageFromArray(im_n.get_fdata())
    im.SetOrigin(-aff[:3,3])
    im.SetSpacing(im_n.header['pixdim'][1:4].tolist())
    
    ##flips image along correct axis according to image properties
    flip_im = sitk.Flip(im, np.diag(aff[:3,:3]<-0).tolist())
    nda = np.rot90(sitk.GetArrayViewFromImage(flip_im))
    return nda.copy()

def get_masses(binary_arr,vol_thresh,intensity_image = None): 
    return [[mass,mass.centroid] for mass in regionprops(spim.label(binary_arr)[0],intensity_image = intensity_image) if mass.area>vol_thresh]

def assign_labels_2_kidneys(k,masses):
    
    if len(masses) == 1:
        closeness_array = np.zeros((len(k)))
        for index in range(len(k)): 
            closeness_array[index] = distance.euclidean(k[index],masses[0])

        masses_to_kidney_association = np.array([np.argmin(closeness_array)])
    else:
        closeness_array = np.zeros((len(k),len(masses)))
        
        for index in range(len(k)): 
            for jdex in range(len(masses)):
                closeness_array[index,jdex] = distance.euclidean(k[index],masses[jdex])
                
        #find each mass's associated kidney, where index in association list corresponds to the tumour's index and
        #the element value corresponds to the kidney's index
        masses_to_kidney_association = np.argmin(closeness_array,axis=0)
        
    return masses_to_kidney_association


def is_sole_kidney_central(kidney_centroids, im,inf, inplane_spac,
                           test1_length = 25, test2_length = 10,
                           axes = None):
    sole_kidney = kidney_centroids[0]
    axial,lr_index,ud = axes
    z_bone,lr_bone,ud_bone = np.array(regionprops((im>250).astype(int))[0].centroid).reshape(-1,1)[np.array(axes)]
    z_bone,lr_bone,ud_bone = z_bone[0],lr_bone[0],ud_bone[0]
    
    # assert(1==2)
    # test 1 - does the centre of the single kidney line up with spine within 25mm? if so - central kidney
    if abs(sole_kidney[lr_index] - lr_bone)*inplane_spac < test1_length:
        return True, ud_bone, lr_bone
    else:
        # test 2 - kidney is also central if wraps around spine.
        # does some portion of the kidney wrap around the spine?
        
        # test 2 distance is 10mm
        _test_extent_inpixels = int((test2_length/2)*inplane_spac)
        
        # create test label - where the pixels within +-10mm of centre of bone attenuating tissue are zeroed out
        _central_test = inf
        _central_test[:,
                      int(lr_bone-_test_extent_inpixels):int(lr_bone+_test_extent_inpixels),
                      :] = 0
        
        # wrapping is true if one or more objects from test label appear either side of the centre of bone attenuating tissue.
        # if wrapping is true, then the kidney is central.
        _test_centroids = [ centroid[lr_index]> lr_bone for _, centroid in get_masses(_central_test,0)]
        if (False in _test_centroids) and (True in _test_centroids):
            return True, ud_bone, lr_bone
        else:
            return False, ud_bone, lr_bone
        
def seg_2_mesh(segmentation,axes=None, show=False):
    axial_index,lr,ud = axes
    if show:
        index = segmentation.shape[axial_index]//2
        if axial_index==1: plt.imshow(segmentation[:,index])
        elif axial_index==2: plt.imshow(segmentation[:,:,index])
        else: plt.imshow(segmentation[index])
    
    verts, faces, norm, val = marching_cubes(segmentation>0, 0.8, step_size=1, allow_degenerate=True)
    if show:
        show_verts = np.round(verts)
        show_verts = show_verts[show_verts[:,axial_index]==index]
        plt.scatter(show_verts[:,lr],show_verts[:,ud])
        plt.show(block=True)
    return np.array([verts,faces],dtype=object)
    
def find_orientation(spacing,kidney_centroids,is_axes=True,im=None):
    rounded_spacing = np.around(spacing,decimals=2)
    assert(2 in np.unique(rounded_spacing,return_counts=True)[1])
    inplane_spacing = stats.mode(rounded_spacing,keepdims=False)[0]
    indices = np.array([0,1,2])
    axial = indices[rounded_spacing!=inplane_spacing][0]
    if is_axes:
        if axial==0: 
            first_half = im[:,:256]
            second_half = im[:,256:]
        else:
            first_half = im[:256]
            second_half = im[256:]
            
        # there should only be one plane of symmetry in a CT scan(roughly): 
        # from the axial perspective that splits the spine in half. Thus, the symmetrical 
        # plane for bones should be up-down. We know the axial plane already, so to determine 
        # up-down, we simplysplit image along the first non-axial slice, and compare bone 
        # totals in each half. if these are roughly similar (within 30% of each other) - we 
        # say this is symmetry, and therefor the up-down plane.
            
        first_total = np.array(regionprops((first_half>250).astype(int))[0].area)
        second_total = np.array(regionprops((second_half>250).astype(int))[0].area)
        fraction = first_total/second_total
        if (fraction>0.7) and (fraction < 1.3):
            if axial==0: lr,ud, = 1,2
            elif axial ==1:lr,ud=0,2
            else: lr,ud=0,1
        else:
            if axial==0: lr,ud, = 2,1
            elif axial ==1:lr,ud=2,0
            else: lr,ud=1,0

        return axial,lr,ud
    # do below when assigning orientations for spacings - where lr and ud distinction doesnt matter
    else:return axial,*indices[rounded_spacing==inplane_spacing][::-1]
            

def create_labelled_dataset(dataset,im_p,infnpy_p,infnii_p,lb_p,save_dir,
                   rawv_p,rawo_p,is_testing=False,size_thresh=200):
    cases = [case for case in os.listdir(im_p) if case.endswith('.nii.gz')]
    cases.sort()
    feature_data = []
    
    for case in cases:
        
        ########### LOAD DATA #############
        print(case)
        inf_n = nib.load(os.path.join(infnii_p,case))
        inf = nifti_2_correctarr(inf_n)
        kid_data = np.array(get_masses(inf>0,20),dtype=object)
        
        im_n = nib.load(os.path.join(im_p,case))
        inf_4mm = np.load(os.path.join(infnpy_p,case[:-7]+'.npy'))
        lb_n = nib.load(os.path.join(lb_p,case))
    
        im = nifti_2_correctarr(im_n)
        lb = nifti_2_correctarr(lb_n)
        
        spacing = inf_n.header['pixdim'][1:4]
        spacing_axes = find_orientation(spacing,kid_data[:,1],is_axes=False)
        z_spac,inplane_spac = spacing[spacing_axes[0]], spacing[spacing_axes[1]]

        axes = find_orientation(im.shape,kid_data[:,1],im=im)
        axial,lr,ud = axes
        vox_volmm = np.prod(spacing)
        
        if axial == 0:inference_centroids = np.asarray([np.asarray([*centroid])*np.array([4/z_spac,4/inplane_spac,4/inplane_spac]) for _,centroid in get_masses(inf_4mm==1,size_thresh)])
        elif axial == 1:inference_centroids = np.asarray([np.asarray([*centroid])*np.array([4/inplane_spac,4/z_spac,4/inplane_spac]) for _,centroid in get_masses(inf_4mm==1,size_thresh)])
        else:inference_centroids = np.asarray([np.asarray([*centroid])*np.array([4/inplane_spac,4/inplane_spac,4/z_spac]) for _,centroid in get_masses(inf_4mm==1,size_thresh)])
        
        inference_statistics = np.asarray([[im.image_filled.sum()*(4**3), im.solidity,im.axis_major_length*4,im.axis_minor_length*4,*im.inertia_tensor_eigvals] for im,_ in get_masses(inf_4mm==1,size_thresh)])
        inference_segmentations = [im.image_filled for im,_ in get_masses(inf_4mm==1,size_thresh)]
        inference_locations = [im.bbox for im,_ in get_masses(inf_4mm>0,size_thresh)]
        inference_intensity = [im.image_intensity for im,_ in get_masses(inf,size_thresh,im)]

        if len(inference_centroids)==1:
            print(case, "has 1 kidney")
            single_kidney_flag=  True
            # check if sole kidney is central, and retrieve centroid of bone-attenuating tissue 
            central_kidney_flag, ud_bone, lr_bone = is_sole_kidney_central(inference_centroids,im,inf,inf_n.header['pixdim'][3], axes=axes)
            if central_kidney_flag:kidneys = ['central']
            elif inference_centroids[0][lr] - lr_bone > 0: kidneys = ['left']
            else:kidneys = ['right']
            print("Sole kidney is in location {}.".format(kidneys[0]))
        else:
            if (len(inference_centroids)==0) or (len(inference_centroids)>2):continue
            # assert(len(inference_centroids)==2)
            single_kidney_flag=  False
            if inference_centroids[0][lr] < inference_centroids[1][lr]: kidneys = ['right','left']
            else: kidneys = ['left','right']
            
        centroids,statistics = [*inference_centroids], [*inference_statistics]
        segmentations = [*inference_segmentations]
        intensities = [*inference_intensity]
        locations = [*inference_locations]
        centre = np.mean(centroids,axis=0)

        if not ((inf.shape[lr]==512) and (inf.shape[ud] == 512)): 
            print("Strange im shape:",inf.shape)
            continue
        
        lb_cancers = np.array([np.array(centroid) for _, centroid in get_masses(lb==2,size_thresh/10)])
        lb_cysts = np.array([np.array(centroid) for _, centroid in get_masses(lb==3,size_thresh/10)])
        
        canc2kid = assign_labels_2_kidneys(centroids,lb_cancers)
        cyst2kid = assign_labels_2_kidneys(centroids,lb_cysts)
        
        cancer_vols = np.asarray([im.area*vox_volmm for im,centroid in get_masses(lb==2,size_thresh/10)])
        cyst_vols = np.asarray([im.area*vox_volmm for im,centroid in get_masses(lb==3,size_thresh/10)])

        obj_meta = np.array([[seg_2_mesh(segmentations[i],axes=axes,show=is_testing),case[:-7]+'_{}'.format(kidneys[i])] for i in range(len(kidneys))],dtype=object)
        objs,names = obj_meta[:,0],obj_meta[:,1].astype(str)
        
        for i,statistic in enumerate(statistics):
            feature_set = feu.generate_features(case,statistic,kidneys[i],i,intensities[i],is_labelled=True,
                                            cancer_vols=cancer_vols,cyst_vols=cyst_vols,canc2kid=canc2kid,cyst2kid=cyst2kid)
            feature_data.append(feature_set)
            

        for obj,name,location in zip(objs,names,locations):
            verts = fu.create_and_save_raw_object(rawv_p,rawo_p,obj,name)
            if is_testing: 
                xmin,ymin,zmin,xmax,ymax,zmax = location
                verts_displaced = np.round(verts+np.array([xmin,ymin,zmin]))
                tu.plot_obj_onlabel(verts_displaced,axes,inf_4mm)

        ########### TESTING #############
        if is_testing: 
            # Printing statistics for testing 
            for i, (vol, convexity, majdim, mindim, _,_,_) in enumerate(statistics):print("{} kidney has a volume of {:.3f}cm cubed.".format(kidneys[i],vol/1000)) 
            for vol,assoc in zip(cancer_vols,canc2kid):print("Cancer has a volume of {:.3f}cm cubed, and belongs to the {} kidney.".format(vol/1000,kidneys[assoc]))
            for vol,assoc in zip(cyst_vols,cyst2kid): print("Cyst has a volume of {:.3f}cm cubed, and belongs to the {} kidney.".format(vol/1000,kidneys[assoc]))
            # Plotting images for testing
            if single_kidney_flag: tu.plot_all_single_kidney(im,centre,centroids[0],kidneys[0],[ud_bone,lr_bone],axes,is_labelled=True,lb_cancers=lb_cancers,lb_cysts=lb_cysts)
            else: tu.plot_all_double_kidney(im,centre,centroids,kidneys,axes,is_labelled=True,lb_cancers=lb_cancers,lb_cysts=lb_cysts)
        print()
    return feature_data

def create_unseen_dataset(dataset,im_p,infnpy_p,infnii_p,save_dir,
                   rawv_p,rawo_p,is_testing=False,size_thresh=200):
    fu.create_folder(save_dir), fu.create_folder(rawv_p),fu.create_folder(rawo_p)

    cases = [case for case in os.listdir(im_p) if case.endswith('.nii.gz')]
    cases.sort()
    feature_data = []
    
    for case in cases:
        ########### LOAD DATA #############
        print(case)
        inf_n = nib.load(os.path.join(infnii_p,case))
        inf = nifti_2_correctarr(inf_n)
        kid_data = np.array(get_masses(inf>0,20),dtype=object)
        
        im_n = nib.load(os.path.join(im_p,case))
        inf_4mm = np.load(os.path.join(infnpy_p,case[:-7]+'.npy'))    
        im = nifti_2_correctarr(im_n)
        
        spacing = inf_n.header['pixdim'][1:4]
        spacing_axes = find_orientation(spacing,kid_data[:,1],is_axes=False)
        z_spac,inplane_spac = spacing[spacing_axes[0]], spacing[spacing_axes[1]]

        axes = find_orientation(im.shape,kid_data[:,1],im=im)
        axial,lr,ud = axes
        
        if axial == 0:inference_centroids = np.asarray([np.asarray([*centroid])*np.array([4/z_spac,4/inplane_spac,4/inplane_spac]) for _,centroid in get_masses(inf_4mm==1,size_thresh)])
        elif axial == 1:inference_centroids = np.asarray([np.asarray([*centroid])*np.array([4/inplane_spac,4/z_spac,4/inplane_spac]) for _,centroid in get_masses(inf_4mm==1,size_thresh)])
        else:inference_centroids = np.asarray([np.asarray([*centroid])*np.array([4/inplane_spac,4/inplane_spac,4/z_spac]) for _,centroid in get_masses(inf_4mm==1,size_thresh)])
        
        inference_statistics = np.asarray([[im.image_filled.sum()*(4**3), im.solidity,im.axis_major_length*4,im.axis_minor_length*4,*im.inertia_tensor_eigvals] for im,_ in get_masses(inf_4mm==1,size_thresh)])
        inference_segmentations = [im.image_filled for im,_ in get_masses(inf_4mm==1,size_thresh)]
        inference_locations = [im.bbox for im,_ in get_masses(inf_4mm==1,size_thresh)]
        inference_intensity = [im.image_intensity for im,_ in get_masses(inf,size_thresh,im)]

        if len(inference_centroids)==1:
            print(case, "has 1 kidney")
            single_kidney_flag=  True
            # check if sole kidney is central, and retrieve centroid of bone-attenuating tissue 
            central_kidney_flag, ud_bone, lr_bone = is_sole_kidney_central(inference_centroids,im,inf,inf_n.header['pixdim'][3], axes=axes)
            if central_kidney_flag:kidneys = ['central']
            elif inference_centroids[0][lr] - lr_bone > 0: kidneys = ['left']
            else:kidneys = ['right']
            print("Sole kidney is in location {}.".format(kidneys[0]))
        else:
            if (len(inference_centroids)==0) or (len(inference_centroids)>2):continue
            # assert(len(inference_centroids)==2)
            single_kidney_flag=  False
            if inference_centroids[0][lr] < inference_centroids[1][lr]: kidneys = ['right','left']
            else: kidneys = ['left','right']
            
        centroids,statistics = [*inference_centroids], [*inference_statistics]
        segmentations = [*inference_segmentations]
        intensities = [*inference_intensity]
        locations = [*inference_locations]
        centre = np.mean(centroids,axis=0)

        if not ((inf.shape[lr]==512) and (inf.shape[ud] == 512)): 
            print("Strange im shape:",inf.shape)
            continue

        for i,statistic in enumerate(statistics):
            feature_set = feu.generate_features(case,statistic,kidneys[i],i,intensities[i],is_labelled=False)
            feature_data.append(feature_set)
            
            
        obj_meta = np.array([[seg_2_mesh(segmentations[i],axes=axes,show=is_testing),case[:-7]+'_{}'.format(kidneys[i])] for i in range(len(kidneys))],dtype=object)
        objs,names = obj_meta[:,0],obj_meta[:,1].astype(str)

        for obj,name,location in zip(objs,names,locations):
            verts = fu.create_and_save_raw_object(rawv_p,rawo_p,obj,name)
            if is_testing: 
                xmin,ymin,zmin,xmax,ymax,zmax = location
                verts_displaced = np.round(verts+np.array([xmin,ymin,zmin]))
                tu.plot_obj_onlabel(verts_displaced,axes,inf_4mm)
                    
        ########### TESTING #############
        if is_testing: 
            # Printing statistics for testing 
            for i, (vol, convexity, majdim, mindim, _,_,_) in enumerate(statistics):print("{} kidney has a volume of {:.3f}cm cubed.".format(kidneys[i],vol/1000)) 
            # Plotting images for testing
            if single_kidney_flag: tu.plot_all_single_kidney(im,centre,centroids[0],kidneys[0],[ud_bone,lr_bone],axes=axes,is_labelled=False)
            else: tu.plot_all_double_kidney(im,centre,centroids,kidneys,axes=axes,is_labelled=False)

        print()
    return feature_data
    

if __name__ == '__main__':

    import pandas as pd
    # is_testing = True shows you statistics and images as you go. test num allows you to choose what case to start at.
    dataset = 'coreg_ncct'
    im_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images/'.format(dataset)
    infnpy_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset)
    infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset)
    lb_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/labels/'
    save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}'.format(dataset)
    rawv_p = os.path.join(save_dir,"raw_vertices")
    rawo_p = os.path.join(save_dir,"raw_objs")
    is_testing=True
    
    feature_data = create_labelled_dataset(dataset,im_p,infnpy_p,infnii_p,lb_p,save_dir,rawv_p,rawo_p,is_testing=is_testing)
    # feature_data = create_unseen_dataset(dataset,im_p,infnpy_p,infnii_p,save_dir,rawv_p,rawo_p,is_testing=is_testing)

    df = pd.DataFrame(feature_data)
    df.to_csv(os.path.join(save_dir,'features_stage1.csv'))
    
    feature_csv_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}/features_stage1.csv'.format(dataset)
    cleaned_objs_path = os.path.join(save_dir,'cleaned_objs')
    curvature_path = os.path.join(save_dir,'curvatures')
    vertices_path = os.path.join(save_dir,'vertices')
    edges_path = os.path.join(save_dir,'edges')
    # df = pd.read_csv(feature_csv_path,index_col=0)
    results = []
    
    for obj in os.listdir(rawo_p):
        entry={}
        if not obj.endswith('.obj'): continue
        nii_case = '_'.join(obj.split('_')[:-1])+'.nii.gz'
        side = obj.split('_')[-1][:-4]
        if side =='central': side = 'centre'
        entry['case'] = nii_case
        entry['position'] = side
        
        obj_file = gmu.smooth_object(obj,rawo_p)
        c,v,e = gmu.extract_object_features(obj_file,obj)
        entry = fu.save_smooth_object_data(entry,c,v,e,obj_file,obj,cleaned_objs_path,
                                 curvature_path,vertices_path,edges_path)
        results.append(entry)
        
    df2 = pd.merge(left=df,right=pd.DataFrame(results),how='outer')
    df2.to_csv(os.path.join(save_dir,'features_stage2.csv'))
