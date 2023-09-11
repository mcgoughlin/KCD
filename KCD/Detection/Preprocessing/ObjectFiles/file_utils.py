import os
import numpy as np
from pandas.api.types import is_numeric_dtype
from stl import mesh
from pymeshfix._meshfix import PyTMesh
# import open3d as o3d
import pandas as pd
import bpy

def create_folder(folder):
    if not os.path.exists(folder):os.mkdir(folder)
    
def convert_stl_obj(stl_loc,obj_loc):
        # Clear all data (OPTIONAL)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Load the STL file into the current Blender scene
    bpy.ops.import_mesh.stl(filepath=stl_loc)
    save_obj(bpy.data.objects[0],obj_loc)
    os.remove(stl_loc)
    
def setup_save_folders(save_dir):
    rawv_p = os.path.join(save_dir,"raw_vertices")
    rawo_p = os.path.join(save_dir,"raw_objs")
    cleano_p = os.path.join(save_dir,'cleaned_objs')
    c_p = os.path.join(save_dir,'curvatures')
    v_p = os.path.join(save_dir,'vertices')
    e_p = os.path.join(save_dir,'edges')
    
    folders = rawv_p,rawo_p,cleano_p,c_p,v_p,e_p
    
    for fold in folders: create_folder(fold)
    
    return folders

def save_obj(mesh_object, filepath):
    with open(filepath, 'w') as f:
        for v in mesh_object.data.vertices:
            f.write("v {0} {1} {2}\n".format(v.co.x, v.co.y, v.co.z))
        for vn in mesh_object.data.vertex_normals:
            x,y,z = vn.vector
            f.write("vn {0} {1} {2}\n".format(x,y,z))
        for i,p in enumerate(mesh_object.data.polygons):
            f.write("f")
            for idx in p.vertices:
                f.write(" {0}//{1}".format(idx + 1,i+1))
            f.write("\n")
            
def save_smooth_object_data(entry,c,v,e,obj,obj_name,
                     co_p,c_p,v_p,e_p):
    for folder in [co_p,c_p,v_p,e_p]: create_folder(folder)
    save_obj(obj,os.path.join(co_p,obj_name))
    np.save(os.path.join(c_p,obj_name[:-4]),c)
    np.save(os.path.join(v_p,obj_name[:-4]),v)
    np.save(os.path.join(e_p,obj_name[:-4]),e)

def create_and_save_raw_object(raw_v_path,raw_obj_path,
                 obj,name):
    mfix = PyTMesh(False)
    verts,faces = obj[0], obj[1]
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    np.save(os.path.join(raw_v_path,name+'.npy'),verts)
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = verts[f[j],:]
    filename = os.path.join(raw_obj_path,name+'.stl')
    cube.save(filename)
    mfix.load_file(filename)
    mfix.fill_small_boundaries(nbe=0, refine=True)
    vert, faces = mfix.return_arrays()
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vert[f[j],:]
    cube.save(filename)
    
    convert_stl_obj(filename,filename[:-4]+".obj")
    # cube = o3d.io.read_triangle_mesh(filename)
    # o3d.io.write_triangle_mesh(filename[:-4]+".obj", cube)
    # os.remove(filename)
    
    # smoothed_object = gmu.smooth_object(filename[:-4]+".obj",raw_obj_path)
    
    return verts

def save_normalisation_params(save_fold:str,overwrite=True,is_labelled=True):
    assert(os.path.exists(save_fold))
    if is_labelled:features_fp = os.path.join(save_fold,'features_labelled.csv')
    else:features_fp = os.path.join(save_fold,'features_unlabelled.csv')
    assert(os.path.exists(features_fp))
    
    params_fp = os.path.join(save_fold,'normalisation_params.csv')
    if os.path.exists(params_fp) and (not overwrite): return
    
    features = pd.read_csv(features_fp)
    
    stats = []
    for col in [column for column in features.columns if (not column.endswith('_vol')) and (is_numeric_dtype(features[column]))]:
        entry = {}
        entry['col']=col
        entry['mean']=features[col].mean()
        entry['std']=features[col].std()
        stats.append(entry)
        
    params = pd.DataFrame(stats)
    params.to_csv(params_fp)