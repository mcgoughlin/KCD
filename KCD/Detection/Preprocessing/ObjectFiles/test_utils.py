#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:43:35 2023

@author: mcgoug01
"""

import matplotlib.pyplot as plt

def plot_single_kidney(im,index,kidney_cent,kidney_label,bone_cent,axes,
                       is_labelled=False,lb_cancers=None,lb_cysts=None):
    
    axial_index,lr_index,ud = axes
    

    if axial_index==2: plt.imshow(im[:,:,index],vmin=-200,vmax=200,cmap='gray')
    elif axial_index==1: plt.imshow(im[:,index],vmin=-200,vmax=200,cmap='gray')
    else: plt.imshow(im[index],vmin=-200,vmax=200,cmap='gray')
    plt.scatter([kidney_cent[lr_index]],[kidney_cent[0]],c='orange',s=20,label = kidney_label)
    plt.scatter([bone_cent[1]],[bone_cent[0]],c='magenta',s=20,marker='*',label='centre of bone')
    if is_labelled:
        plt.scatter(lb_cancers[:,lr_index],lb_cancers[:,0], c = 'cyan',s=5, label='cancer')
        if len(lb_cysts)>0:
            plt.scatter(lb_cysts[:,lr_index],lb_cysts[:,0], c = 'green',s=5, label='cyst')
    plt.axis('off')
    
def plot_double_kidney(im,index,centroids,kidneys,axes,
                       is_labelled=False,lb_cancers=None,lb_cysts=None):
    
    axial_index,lr_index,ud = axes

    if axial_index==2: plt.imshow(im[:,:,index],vmin=-200,vmax=200,cmap='gray')
    elif axial_index==1: plt.imshow(im[:,index],vmin=-200,vmax=200,cmap='gray')
    else: plt.imshow(im[index],vmin=-200,vmax=200,cmap='gray')
    
        
    plt.scatter([centroids[0][lr_index]],[centroids[0][ud]],c='red',s=20, label = 'raw im centroid {}'.format(kidneys[0]))
    plt.scatter([centroids[1][lr_index]],[centroids[1][ud]],c='blue',s=20, label = 'raw im centroid {}'.format(kidneys[1]))
    if is_labelled:
        plt.scatter(lb_cancers[:,lr_index],lb_cancers[:,0], c = 'cyan',s=5, label='cancer')
        if len(lb_cysts)>0:
            plt.scatter(lb_cysts[:,lr_index],lb_cysts[:,0], c = 'green',s=5, label='cyst')
    plt.axis('off')
    
def plot_all_single_kidney(im, centre, kidney_centre, 
                           kidney_label, bone_cent, axes,
                           is_labelled=False,lb_cancers=None,lb_cysts=None):
    
    plt.figure(figsize = (8,8))
    plt.subplot(221)
    axial_index,lr,ud = axes
    plot_single_kidney(im,int(centre[axial_index])-10,kidney_centre,kidney_label,
                       bone_cent,axes,is_labelled=is_labelled,lb_cancers=lb_cancers,lb_cysts=lb_cysts)

    plt.subplot(222)
    plot_single_kidney(im,int(centre[axial_index]),kidney_centre,kidney_label,
                       bone_cent,axes,is_labelled=is_labelled,lb_cancers=lb_cancers,lb_cysts=lb_cysts)
    plt.legend()

    plt.subplot(223)
    plot_single_kidney(im,int(centre[axial_index])+10,kidney_centre,kidney_label,
                       bone_cent,axes,is_labelled=is_labelled,lb_cancers=lb_cancers,lb_cysts=lb_cysts)

    plt.subplot(224)
    plot_single_kidney(im,0,kidney_centre,kidney_label,
                       bone_cent,axes,is_labelled=is_labelled,lb_cancers=lb_cancers,lb_cysts=lb_cysts)
    plt.show(block=True)
    
def plot_all_double_kidney(im,centre,centroids,kidneys,axes,
                           is_labelled=False,lb_cancers=None,lb_cysts=None):

    
    axial_index,lr,ud = axes
    
    plt.figure(figsize = (8,8))
    plt.subplot(221)
    plot_double_kidney(im,int(centre[axial_index]-10),centroids,kidneys,axes,
                       is_labelled=is_labelled,lb_cancers=lb_cancers,lb_cysts=lb_cysts)

    plt.subplot(222)
    plot_double_kidney(im,int(centre[axial_index]),centroids,kidneys,axes,
                       is_labelled=is_labelled,lb_cancers=lb_cancers,lb_cysts=lb_cysts)
    plt.legend()

    plt.subplot(223)
    plot_double_kidney(im,int(centre[axial_index]+10),centroids,kidneys,axes,
                       is_labelled=is_labelled,lb_cancers=lb_cancers,lb_cysts=lb_cysts)

    plt.subplot(224)
    plot_double_kidney(im,0,centroids,kidneys,axes,
                       is_labelled=is_labelled,lb_cancers=lb_cancers,lb_cysts=lb_cysts)
    plt.show(block=True)
    
def plot_obj_onlabel(verts_displaced,axes,inf_4mm):
    axial,lr,ud=axes
    index = int(verts_displaced[:,axial].mean())
    verts_displaced = verts_displaced[verts_displaced[:,axial]==index]
    if axial==0:plt.imshow(inf_4mm[index])
    elif axial==1:plt.imshow(inf_4mm[:,index])
    else:plt.imshow(inf_4mm[:,:,index])
    plt.scatter(verts_displaced[:,lr],verts_displaced[:,ud])
    plt.show(block=True)