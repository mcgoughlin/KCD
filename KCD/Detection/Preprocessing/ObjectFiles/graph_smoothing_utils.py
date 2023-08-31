# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:42:49 2023

@author: mcgoug01
"""

import bpy
import bmesh
from mathutils import Vector
from math import sqrt, asin
import os
import numpy as np
import pandas as pd
import file_utils as fu

# Helper function to select vertices
def select_vertices( obj, indices, end_with_mode ):
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    for index in indices:
        obj.data.vertices[index].select = True
    bpy.ops.object.mode_set(mode = end_with_mode) 

def ensure_vertex_group( obj, group_name ):
    vertex_group = obj.vertex_groups.get( group_name )
    if not vertex_group:
        vertex_group = obj.vertex_groups.new( name = group_name )
    for i in range( len(obj.data.vertices) ):
        vertex_group.add( [i], 0, 'ADD' )
    return vertex_group

def assign_to_vertex_group( obj, group_name, curvatures ):
    vertex_group = ensure_vertex_group( obj, group_name )

    curvatures = [abs(c) for c in curvatures]

    min_curvature = min( curvatures )
    max_curvature = max( curvatures )
    vg_fac = 1.0 / (max_curvature - min_curvature) if max_curvature != min_curvature else 1.0

    for i, vert in enumerate( obj.data.vertices ):
        vertex_group.add( [vert.index], (curvatures[i] - min_curvature) * vg_fac, 'REPLACE' )

def index_of( element, sequence ):
    for i, e in enumerate( sequence ):
        if e == element: return i
    return -1

def search_link( value, links, position ):
    for l in links:
        if l[position] == value: return l
    return None

def rotate( l, n ):
    return l[n:] + l[:n]

# Get vertices in the face order but starting from a given vert
def following_verts_of_vert( vert, face ):
    i0 = index_of( vert, face.verts )
    i1 = (i0 + 1) % 3
    i2 = (i0 + 2) % 3
    return face.verts[i0], face.verts[i1], face.verts[i2]

# Create the oriented ring around vert
def ring_from_vert( vert ):
    vertices = []
    for face in vert.link_faces:
        i0, i1, i2 = following_verts_of_vert( vert, face )
        vertices.append( [i1, i2] )
    result = vertices[0]    
    added = True
    while added and len(vertices):
        added = False
        prev = search_link( result[0], vertices, 1 )
        if prev:
            result = [prev[0]] + result
            vertices.remove( prev )
            added = True
        next = search_link( result[-1], vertices, 0 )
        if next and next[1] not in result:
            result.append( next[1] )
            vertices.remove( next )
            added = True
    return result

def curvature_along_edge( vert, other ):
    normal_diff = other.normal - vert.normal
    vert_diff = other.co - vert.co
    return normal_diff.dot( vert_diff ) / (vert_diff.length_squared+1e-6)

def angle_between_edges( vert, other1, other2 ):
    edge1 = other1.co - vert.co
    edge2 = other2.co - vert.co
    product = edge1.cross( edge2 )
    sinus = product.length / ((edge1.length * edge2.length)+1e-6)
    return asin( min(1.0, sinus) )

def mean_curvature_vert( vert ):
    ring = ring_from_vert( vert )
    ring_curvatures = [curvature_along_edge( vert, other ) for other in ring]
    total_angle = 0.000001
    curvature = 0.0
    for i in range(len(ring)-1):
        angle = angle_between_edges( vert, ring[i], ring[i+1] )
        total_angle += angle
        curvature += angle * (ring_curvatures[i] + ring_curvatures[i+1])
    
    return curvature / (2.0 * total_angle)

def mean_curvature( obj ):

    # Get bmesh access on the mesh
    bm = bmesh.new()
    bm.from_mesh( obj.data )

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    curvatures = []
    for vert in bm.verts:
        curvature = mean_curvature_vert( vert )
        curvatures.append( curvature )
    
    return np.asarray(curvatures)

def get_hist(data,bins=10,range_ = (0,1)):
    binned,names = np.histogram(data,bins=bins,range = range_,density= True)
    return binned, names

def smooth_object(obj,obj_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.obj(filepath=os.path.join(obj_path,obj))
    bpy.context.view_layer.objects.active = bpy.data.objects[0]

    bpy.ops.object.modifier_add(type='REMESH')
    bpy.context.object.modifiers["Remesh"].mode = "VOXEL"
    bpy.context.object.modifiers["Remesh"].voxel_size = 1.2
    bpy.ops.object.modifier_apply(modifier="Remesh")
    
    bpy.ops.object.modifier_add(type='SMOOTH')
    bpy.context.object.modifiers["Smooth"].factor = 0.5
    bpy.context.object.modifiers["Smooth"].iterations = 5
    bpy.ops.object.modifier_apply(modifier="Smooth")
    return bpy.data.objects[0]

def extract_object_features(obj,name):
    vertices = np.asarray([[vert.co.x, vert.co.y, vert.co.z] for vert in obj.data.vertices])
    edges = np.asarray([[vert for vert in polygon.vertices] for polygon in obj.data.edges])
    curvatures= mean_curvature( obj)
    return curvatures, vertices,edges


if __name__ == "__main__":
        
    dataset = 'coreg_ncct'
    obj_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}/raw_objs'.format(dataset)
    save_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}'.format(dataset)
    feature_csv_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Classification/object_dataset/{}/features_stage1.csv'.format(dataset)
    cleaned_objs_path = os.path.join(save_path,'cleaned_objs')
    curvature_path = os.path.join(save_path,'curvatures')
    vertices_path = os.path.join(save_path,'vertices')
    edges_path = os.path.join(save_path,'edges')
    # df = pd.read_csv(feature_csv_path,index_col=0)
    results = []
    
    for obj in os.listdir(obj_path):
        entry={}
        if not obj.endswith('.obj'): continue
        nii_case = '_'.join(obj.split('_')[:-1])+'.nii.gz'
        side = obj.split('_')[-1][:-4]
        if side =='central': side = 'centre'
        entry['case'] = nii_case
        entry['position'] = side
        
        obj_file = smooth_object(obj,obj_path)
        c,v,e = extract_object_features(obj_file,obj)
        entry = fu.save_object_data(entry,c,v,e,obj_file,obj,cleaned_objs_path,
                                 curvature_path,vertices_path,edges_path)
        results.append(entry)
        # df2 = pd.merge(left=df,right=pd.DataFrame(results),how='outer')
        # df2.to_csv(os.path.join(save_path,'features_stage2.csv'))


