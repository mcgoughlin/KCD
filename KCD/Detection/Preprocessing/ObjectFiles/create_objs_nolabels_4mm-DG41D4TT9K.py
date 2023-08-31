

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.spatial import distance

from skimage.measure import regionprops,marching_cubes
import scipy.ndimage as spim
from stl import mesh
from pymeshfix._meshfix import PyTMesh
import open3d as o3d


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


def get_hist(intensity_data):
    binned,names = np.histogram(intensity_data,range = (-20,80),density= True)
    return binned, names

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

def assign_lrindex(is_kits19):
    if is_kits19:
        return 1
    else:
        return 2

def plot_single_kidney(im,index,kidney_cent,kidney_label,
                       bone_cent,cent_4mm,is_kits19=True):
    
    lr_index=1

    if lr_index==1:
        plt.imshow(im[:,:,index],vmin=-100,vmax=200,cmap='gray')
    else:
        plt.imshow(im[:,index],vmin=-100,vmax=200,cmap='gray')
    plt.scatter([kidney_cent[lr_index]],[kidney_cent[0]],c='orange',s=20,label = kidney_label)
    plt.scatter([bone_cent[1]],[bone_cent[0]],c='magenta',s=20,marker='*',label='centre of bone')
    plt.scatter([cent_4mm[lr_index]],[cent_4mm[0]],c='orange',s=30,marker='v', label = '4mm centroid')
    plt.axis('off')
    
def plot_double_kidney(im,index,right_kidney,left_kidney,
                       left_4mm,right_4mm,is_kits19=True):
    
    lr_index = 1

    if lr_index==1:
        plt.imshow(im[:,:,index],vmin=-100,vmax=200,cmap='gray')
    else:
        plt.imshow(im[:,index],vmin=-100,vmax=200,cmap='gray')
        
    print(left_kidney,right_kidney)

    plt.scatter([right_kidney[lr_index]],[right_kidney[0]],c='red',s=20, label = 'raw im centroid right')
    plt.scatter([left_kidney[lr_index]],[left_kidney[0]],c='blue',s=20, label = 'raw im centroid left')
    plt.scatter([right_4mm[lr_index]],[right_4mm[0]],c='red',s=30,marker='v', label = '4mm centroid right')
    plt.scatter([left_4mm[lr_index]],[left_4mm[0]],c='blue',s=30,marker='v', label = '4mm centroid left')
    plt.axis('off')
    
def plot_all_single_kidney(im, centre, kidney_centre, 
                           kidney_label, bone_cent, 
                           cent_4mm,is_kits19=True):
    
    plt.figure(figsize = (8,8))
    plt.subplot(221)
    if is_kits19:axial_index = 2
    else:axial_index=1
    plot_single_kidney(im,int(centre[axial_index])-10,kidney_centre,'central_kidney',
                       bone_cent,cent_4mm,is_kits19=is_kits19)

    plt.subplot(222)
    plot_single_kidney(im,int(centre[axial_index]),kidney_centre,'central_kidney',
                       bone_cent,cent_4mm,is_kits19=is_kits19)
    plt.legend()

    plt.subplot(223)
    plot_single_kidney(im,int(centre[axial_index])+10,kidney_centre,'central_kidney',
                       bone_cent,cent_4mm,is_kits19=is_kits19)

    plt.subplot(224)
    plot_single_kidney(im,0,kidney_centre,'central_kidney',
                       bone_cent,cent_4mm,is_kits19=is_kits19)
    plt.show(block=True)
    
def plot_all_double_kidney(im,centre,right_kidney,
                   left_kidney,left_4mm,right_4mm,
                   dataset='kits_ncct'):

    is_kits19 = (dataset=='kits_ncct')
    
    axial_index = 2
    
    plt.figure(figsize = (8,8))
    plt.subplot(221)
    plot_double_kidney(im,int(centre[axial_index]-10),right_kidney,
                       left_kidney,left_4mm,right_4mm,
                       is_kits19=is_kits19)

    plt.subplot(222)
    plot_double_kidney(im,int(centre[axial_index]),right_kidney,
                       left_kidney,left_4mm,right_4mm,
                       is_kits19=is_kits19)
    plt.legend()

    plt.subplot(223)
    plot_double_kidney(im,int(centre[axial_index]+10),right_kidney,
                       left_kidney,left_4mm,right_4mm,
                       is_kits19=is_kits19)

    plt.subplot(224)
    plot_double_kidney(im,0,right_kidney,
                       left_kidney,left_4mm,right_4mm,
                       is_kits19=is_kits19)
    plt.show(block=True)
    
def is_sole_kidney_central(kidney_centroids, im,inf, inplane_spac,
                           test1_length = 25, test2_length = 10,
                           is_kits19=True,is_coreg=False):
    sole_kidney = kidney_centroids[0]
    lr_index = assign_lrindex(is_kits19)

    # Find centre of bone-attenuating tissue - lr is left-to-right on axial, ud is up-down
    if is_kits19:
        ud_bone,lr_bone,z_bone = regionprops((im>250).astype(int))[0].centroid
    else:
        ud_bone,z_bone,lr_bone = regionprops((im>250).astype(int))[0].centroid
        
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
        
def seg_2_mesh(segmentation,dataset='kits_ncct', show=True):
    if (dataset=='kits_ncct') or (dataset=='add_ncct_unseen'):
        index = int(segmentation.shape[-1]/2)
        if show:
            plt.imshow(segmentation[:,:,index])
    else:
        index = int(segmentation.shape[1]/2)
        if show:
            plt.imshow(segmentation[:,index])
    
    verts, faces, norm, val = marching_cubes(segmentation>0, 0.8, step_size=1, allow_degenerate=True)
    if show:
        show_verts = np.round(verts)
        if (dataset=='kits_ncct') or (dataset=='add_ncct_unseen'):
            show_verts = show_verts[show_verts[:,2]==index]
            plt.scatter(show_verts[:,1],show_verts[:,0])
        else:
            show_verts = show_verts[show_verts[:,1]==index]
            plt.scatter(show_verts[:,2],show_verts[:,0])
            
        plt.show(block=True)
    return np.array([verts,faces],dtype=object)
    


if __name__ == '__main__':

    
    # is_testing = True shows you stats and images as you go. test num allows you to choose what case to start at.
    is_testing = True
    test_num = 48

    # True if creating dataset from ncct images, false if sncct
    # is_kits19ncct = True
    
    datasets = ['add_ncct_unseen']

    for dataset in datasets:
        
        if dataset =='kits_ncct':
            is_kits19ncct=True
        else:
            is_kits19ncct=False
        
        # set path variables
        im_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/{}/images/'.format(dataset)
        infnpy_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_npy/{}/[4 4 4]mm/'.format(dataset)
        if (dataset=='coreg_ncct'):
            infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset)

        else:
            infnii_p = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/predictions_nii/{}/[4 4 4]mm/'.format(dataset)

        save_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}'.format(dataset)
        
        cases = [case for case in os.listdir(im_p) if case.endswith('.nii.gz')]
        cases.sort()
        
        feature_data = []
        
        for case in cases:
                        
            if is_kits19ncct:
                case_num = int(case.split('-')[1][:5])
            else:
                if (dataset=='coreg_ncct'):
                    case_num = int(case.split('_')[1][:3])
                elif (dataset=='kits_ncct_labelled'):
                    case_num = int(case.split('-')[1][:5])
                else:
                    case_num = int(case.split('_')[1][:3])
    
            if not (case_num>=test_num): continue
            
        ########### LOAD DATA #############
    
            print(case)
            im_n = nib.load(os.path.join(im_p,case))
            inf_n = nib.load(os.path.join(infnii_p,case))
            inf_4mm = np.load(os.path.join(infnpy_p,case[:-7]+'.npy'))
            print(inf_4mm.shape)
            
            try:
                im = nifti_2_correctarr(im_n)
                inf = nifti_2_correctarr(inf_n)
            except:
                print(case,'######################')

                continue
            if is_kits19ncct or (dataset=='add_ncct_unseen'):
                z_spac = inf_n.header['pixdim'][3]
                inplane_spac = np.mean(np.abs(inf_n.header['pixdim'][1:3]))
                label_spacing = np.prod(inf_n.header['pixdim'][1:4])
            else:
                z_spac = inf_n.header['pixdim'][1]
                inplane_spac = np.mean(np.abs(inf_n.header['pixdim'][2:4]))
                label_spacing = np.prod(inf_n.header['pixdim'][1:4])
                
            vox_volmm = inplane_spac*inplane_spac*z_spac   
            
        ########### GENERATE FEATURES #############
        
            if is_kits19ncct or (dataset=='add_ncct_unseen'):
                inference_centroids = np.asarray([np.asarray([*centroid])*np.array([4/inplane_spac,4/inplane_spac,4/z_spac]) for _,centroid in get_masses(inf_4mm==1,200)])
            else:
                inference_centroids = np.asarray([np.asarray([*centroid])*np.array([4/inplane_spac,4/z_spac,4/inplane_spac]) for _,centroid in get_masses(inf_4mm==1,200)])
                
            # assert(1==2)
            # Extract segmentation inference statistics - correct for preprocessing voxel size (4mm)
            # solidity is just convexity. inertia tensor eigenvalues gives you a sense of the orientation of the mass
            inference_stats = np.asarray([[im.image_filled.sum()*(4**3), im.solidity,im.axis_major_length*4,im.axis_minor_length*4,*im.inertia_tensor_eigvals] for im,_ in get_masses(inf_4mm==1,200)])
            inference_segmentations = [im.image_filled for im,_ in get_masses(inf_4mm==1,200)]
            inference_locations = [im.bbox for im,_ in get_masses(inf_4mm==1,200)]
            inference_intensity = [im.image_intensity for im,_ in get_masses(inf,200,im)]
            
            lr_index = assign_lrindex(is_kits19ncct or (dataset=='add_ncct_unseen'))
    
            if len(inference_centroids)==1:
                print(case, "has 1 kidney")
                single_kidney_flag=  True
                    
                # check if sole kidney is central, and retrieve centroid of bone-attenuating tissue 
                central_kidney_flag, ud_bone, lr_bone = is_sole_kidney_central(inference_centroids,im,inf,
                                                                               inf_n.header['pixdim'][3], is_kits19=is_kits19ncct or (dataset=='add_ncct_unseen'))
    
                
                if central_kidney_flag:
                    print("Sole kidney is central.")
                    central_kidney = inference_centroids[0]
                    cent_4mm,cent_stats = inference_centroids[0], inference_stats[0]
                    central_segmentation = inference_segmentations[0]
                    central_intensity = inference_intensity[0]
                    central_location = inference_locations[0]
                    right_kidney,left_kidney,left_stats,right_stats,right_segmentation,left_segmentation,left_intensity,right_intensity = [None]*8
                    centre = np.array([*central_kidney])
                elif inference_centroids[0][lr_index] - lr_bone > 0:
                    print("Sole kidney is on the left.")
                    central_kidney_flag = False
                    left_kidney = inference_centroids[0]
                    left_segmentation = inference_segmentations[0]
                    left_intensity = inference_intensity[0]
                    left_location = inference_locations[0]
                    left_4mm,left_stats = inference_centroids[0], inference_stats[0]
                    right_kidney,right_stats,right_segmentation,right_intensity = None,None,None,None
                    centre = np.array([*left_kidney])
        
                else:
                    print("Sole kidney is on the right.")
                    central_kidney_flag = False
                    right_kidney = inference_centroids[0]
                    right_intensity = inference_intensity[0]
                    right_location= inference_locations[0]
                    right_segmentation = inference_segmentations[0]
                    right_4mm,right_stats = inference_centroids[0], inference_stats[0]
                    left_kidney,left_stats,left_segmentation, left_intensity = None,None,None,None
                    centre = np.array([*right_kidney])
            else:
                if (len(inference_centroids)==0) or (len(inference_centroids)>2):continue
                # assert(len(inference_centroids)==2)
                single_kidney_flag=  False
                central_kidney_flag = False
        
                first_kidney = inference_centroids[0]
                second_kidney = inference_centroids[1]
            
                if first_kidney[lr_index] < second_kidney[lr_index]:
                    right_kidney = first_kidney
                    left_kidney = second_kidney
                    right_index = 0
                    left_index = 1
                else:
                    right_kidney = second_kidney
                    left_kidney = first_kidney
                    right_index = 1
                    left_index=0
                right_4mm,right_stats,right_segmentation, right_intensity = inference_centroids[right_index], inference_stats[right_index],inference_segmentations[right_index],inference_intensity[right_index]
                left_4mm,left_stats,left_segmentation,left_intensity = inference_centroids[left_index], inference_stats[left_index],inference_segmentations[left_index], inference_intensity[left_index]
                right_location,left_location = inference_locations[right_index], inference_locations[left_index]
                centre = np.mean([left_kidney,right_kidney],axis=0)
            
            # z-axis appears as the middle dimension - so weird! but this is fine - just need to check this is consistent in all labels
            if  (dataset=='add_ncct_unseen'):
                if not ((inf.shape[0]==512) and (inf.shape[1] == 512)): 
                    print("Strange im shape:",inf.shape)
                    print(im_n.get_fdata().shape)
                    print(im_n.header['pixdim'][1:4])
                    print()
                    continue
        

            
            
            # kidney-centric data collection
            # im.image_filled.sum()*(4**3), im.solidity,im.axis_major_length*4,im.axis_minor_length*4,*im.inertia_tensor_eigvals
            if single_kidney_flag:
                if central_kidney_flag:
                    for i,inf in enumerate([cent_stats]):
                        entry = {}
                        entry['case'] = case
                        entry['volume'] = inf[0]
                        entry['convexity'] = inf[1]
                        entry['maj_dim'] = inf[2]
                        entry['min_dim'] = inf[3]
                        entry['eigvec1'] = inf[4]
                        entry['eigvec2'] = inf[5]
                        entry['eigvec3'] = inf[6]
                        entry['position'] = 'centre'
                            
                        for bin_, name in zip(*get_hist(central_intensity)):
                            name = 'intens'+str(name)
                            entry[name] = bin_
                            
                        feature_data.append(entry)
                elif type(left_kidney) == type(None):
                    for i,inf in enumerate([right_stats]):
                        entry = {}
                        entry['case'] = case
                        entry['volume'] = inf[0]
                        entry['convexity'] = inf[1]
                        entry['maj_dim'] = inf[2]
                        entry['min_dim'] = inf[3]
                        entry['eigvec1'] = inf[4]
                        entry['eigvec2'] = inf[5]
                        entry['eigvec3'] = inf[6]
                        entry['position'] = 'right'
                        for bin_, name in zip(*get_hist(right_intensity)):
                            name = 'intens'+str(name)
                            entry[name] = bin_
                        feature_data.append(entry) 
                        
                else:
                    for i,inf in enumerate([left_stats]):
                        entry = {}
                        entry['case'] = case
                        entry['volume'] = inf[0]
                        entry['convexity'] = inf[1]
                        entry['maj_dim'] = inf[2]
                        entry['min_dim'] = inf[3]
                        entry['eigvec1'] = inf[4]
                        entry['eigvec2'] = inf[5]
                        entry['eigvec3'] = inf[6]
                        entry['position'] = 'left'
                            
                        for bin_, name in zip(*get_hist(left_intensity)):
                            name = 'intens'+str(name)
                            entry[name] = bin_
                        feature_data.append(entry) 
                        
            else:
                for i,(intens,inf) in enumerate(zip([left_intensity,right_intensity],[left_stats,right_stats])):
                    entry = {}
                    entry['case'] = case
                    entry['volume'] = inf[0]
                    entry['convexity'] = inf[1]
                    entry['maj_dim'] = inf[2]
                    entry['min_dim'] = inf[3]
                    entry['eigvec1'] = inf[4]
                    entry['eigvec2'] = inf[5]
                    entry['eigvec3'] = inf[6]
                    
                    for bin_, name in zip(*get_hist(intens)):
                        name = 'intens'+str(name)
                        entry[name] = bin_
                    
                    if i==1:
                        # current kidney we are collecting data for is right side
                        is_right = True
                        entry['position'] = 'right'
                    else:
    
                        entry['position'] = 'left'
                                
                    feature_data.append(entry)
                
            
        ########### GENERATE OBJECT FILES #############
            if single_kidney_flag:
                if central_kidney_flag:
                    central_obj = seg_2_mesh(central_segmentation,dataset=dataset,show=is_testing)
                    objs = [central_obj]
                    locations = [central_location]
                    names = [case[:-7]+'_centre']
                else:
                    if type(left_kidney) == type(None):
                        right_obj = seg_2_mesh(right_segmentation,dataset=dataset,show=is_testing)
                        names = [case[:-7]+'_right']
                        objs = [right_obj]
                        locations = [right_location]
    
                    else:
                        left_obj = seg_2_mesh(left_segmentation,dataset=dataset,show=is_testing)
                        names = [case[:-7]+'_left']
                        objs = [left_obj]
                        locations = [left_location]
    
            else:
                right_obj = seg_2_mesh(right_segmentation,dataset=dataset,show=is_testing)
                left_obj = seg_2_mesh(left_segmentation,dataset=dataset,show=is_testing)
                objs= [left_obj,right_obj]
                locations = [left_location,right_location]
    
                names = [case[:-7]+'_left',case[:-7]+'_right']
    
            for obj,name,location in zip(objs,names,locations):
                xmin,ymin,zmin,xmax,ymax,zmax = location
                mfix = PyTMesh(False)
                verts = obj[0]
                np.save(os.path.join(save_dir,"raw_vertices_4mm",name+'.npy'),verts)
                faces = obj[1]
                cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        cube.vectors[i][j] = verts[f[j],:]
                filename = os.path.join(save_dir,"raw_objs_4mm",name+'.stl')
                cube.save(filename)
                mfix.load_file(filename)
                mfix.fill_small_boundaries(nbe=0, refine=True)
                vert, faces = mfix.return_arrays()
                cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        cube.vectors[i][j] = vert[f[j],:]
                cube.save(filename)
                cube = o3d.io.read_triangle_mesh(filename)
                o3d.io.write_triangle_mesh(filename[:-4]+".obj", cube)
                os.remove(filename)
                
                verts_displaced = np.round(verts+np.array([xmin,ymin,zmin]))
                if is_testing:
                    if (dataset=='kits_ncct') or (dataset=='add_ncct_unseen'):
                        index = int(zmin + verts[:,2].max()/2)
                        verts_displaced = verts_displaced[verts_displaced[:,2]==index]
                        plt.imshow(inf_4mm[:,:,index])
                        plt.scatter(verts_displaced[:,1],verts_displaced[:,0])
                        plt.show(block=True)
                    else:
                        index = int(ymin + verts[:,1].max()/2)
                        verts_displaced = verts_displaced[verts_displaced[:,1]==index]
                        plt.imshow(inf_4mm[:,index])
                        plt.scatter(verts_displaced[:,2],verts_displaced[:,0])
                        plt.show(block=True)
            
        ########### TESTING #############
        
            
            if is_testing: 
                # Printing stats for testing 
                if single_kidney_flag:
                    if central_kidney_flag:
                        for i, (vol, convexity, majdim, mindim, _,_,_) in enumerate([cent_stats]):
                            string = 'Central'
                            print("{} kidney has a volume of {:.3f}cm cubed.".format(string,vol/1000))
                    else:
                        if type(left_kidney) == type(None):
                            for i, (vol, convexity, majdim, mindim, _,_,_) in enumerate([right_stats]):
                                string = 'Right'
                                print("{} kidney has a volume of {:.3f}cm cubed.".format(string,vol/1000))
                        else:
                            for i, (vol, convexity, majdim, mindim, _,_,_) in enumerate([left_stats]):
                                string = 'Left'
                                print("{} kidney has a volume of {:.3f}cm cubed.".format(string,vol/1000))
                else:
                    for i, (vol, convexity, majdim, mindim, _,_,_) in enumerate([left_stats,right_stats]):
                        if i ==1: string = 'Right'
                        else: string = 'Left'
                        print("{} kidney has a volume of {:.3f}cm cubed.".format(string,vol/1000))
                    
                # Plotting images for testing
                if single_kidney_flag:
                    if central_kidney_flag:
                        plot_all_single_kidney(im,centre,central_kidney,'central_kidney',
                                           [ud_bone,lr_bone],cent_4mm,
                                           is_kits19=is_kits19ncct or (dataset=='coreg_ncct'))
                    else:
                        if type(left_kidney) == type(None):
                            plot_all_single_kidney(im,centre,right_kidney,'left_kidney',
                                               [ud_bone,lr_bone],right_4mm,
                                               is_kits19=is_kits19ncct or (dataset=='coreg_ncct'))
                        else:
                            plot_all_single_kidney(im,centre,left_kidney,'right_kidney',
                                               [ud_bone,lr_bone],left_4mm,
                                               is_kits19=is_kits19ncct or (dataset=='coreg_ncct'))
                            
                else:
                    plot_all_double_kidney(im,centre,right_kidney,
                                       left_kidney,left_4mm,right_4mm,
                                       dataset=dataset)
                
                print()
        
        import pandas as pd
        
        df = pd.DataFrame(feature_data)
        df.to_csv(os.path.join(save_dir,'features_stage1_4mm.csv'))
            