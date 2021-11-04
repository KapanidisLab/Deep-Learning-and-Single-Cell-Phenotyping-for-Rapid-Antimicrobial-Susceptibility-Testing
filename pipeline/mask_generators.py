# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:50:28 2020

@author: Aleksander Zagajewski
"""

from helpers import *
from tqdm import tqdm
import numpy as np
import skimage
import os


def masks_from_VOTT(**kwargs):

    import json, numpy, skimage.io,skimage.draw, sys

    '''
    Locates the VOTT json file in the mask folder and generates single cell masks in the output folder, in the structure
    output_folder/annots/(image_name)/Cell1.png... Cell2.png....

    Parameters
    ----------
    mask_path : string
        path to directory with annotations
    output_path : string
        path to directory where the output data structure will be created.
        Function creates a folder called annots. Inside annots, each subdir is a separate image, inside which are binary masks.

    Returns
    -------

    '''

    mask_path = kwargs.get('mask_path', False)
    output_path = kwargs.get('output_path', False)

    if not all([mask_path, output_path]):  # Verify input
        raise TypeError


    tracker = 0
    for root, dirs, files in tqdm(os.walk(mask_path, topdown=True), total = dircounter(mask_path), unit = 'files', desc = 'Searching directory for export file.'):
        for file in files:
            if file.endswith('export.json'):
                tracker = tracker + 1
                export_file = os.path.join(root, file)

    assert tracker == 1, 'Warning - either no export file found, or multiple copies exist in given folder. Aborting.'

    makedir(os.path.join(output_path, 'annots')) #Create folder with masks

    with open(export_file) as json_file: #Navigate through json structure and extract information
        datafile = json.load(json_file)
        assets = datafile['assets']


        image_total = 0 #Counter for total images and masks written
        mask_total = 0

        for image_key in assets:
            image = assets[image_key]
            image_metadata = image['asset']
            image_filename, image_size = image_metadata['name'], image_metadata['size']
            image_size = (image_size['height'], image_size['width'])

            image_filename = image_filename.split('.')[0] #Remove file extension

            regions = image['regions']

            makedir(os.path.join(output_path,'annots',image_filename)) #Create image subdirectory

            for cellcount,ROI in enumerate(regions):

                if ROI['type'] != 'POLYGON':
                    continue #Exclude non polygon ROIs
                if ROI['boundingBox']['height'] == 0 or ROI['boundingBox']['width'] == 0:
                    continue #Exclude straight lines

                points = ROI['points']
                verts = numpy.zeros((points.__len__(), 2))

                for counter, point in enumerate(points):
                    verts[counter, 0] = point['y'] #Extract polygon verts
                    verts[counter, 1] = point['x']

                mask = skimage.draw.polygon2mask(image_size, verts)
                mask = skimage.img_as_ubyte(mask)

                filename = 'Cell'+str(cellcount)+'.bmp' #Mask filename
                savepath = os.path.join(output_path,'annots',image_filename,filename) #Assemble whole save path
                skimage.io.imsave(savepath,mask, check_contrast=False)

                mask_total = mask_total + 1
            image_total = image_total + 1

    #At the end, print a summary

    sys.stdout.flush()
    print('')
    print('Generated', ':', str(mask_total), 'masks out of', str(image_total), 'images.')
    sys.stdout.flush()


def cellpose_file_loader(maskpath, image_size=None, completeload=False):
    '''

    Args:
        maskpath: (str) path to file with masks
        image_size: (tuple of int) size of image corresponding to maks.
        completeload: (bool) if True, loader will load complete record. In CELLPOSE loader, only False is supported.

    Returns: tuple (masks,model) where masks is a (x,y,N) bool ndarray.

    WARNING - CELLPOSE saves results as raw pickles, which introduces a security vulnerability.

    '''

    data = np.load(maskpath, allow_pickle=True).item()
    masks = data["masks"]

    mask_idxs = np.unique(masks)

    #Remove degenerate case - the zero
    if 0 in mask_idxs:
        i = np.where(mask_idxs==0)
        mask_idxs = np.delete(mask_idxs,i)

    output = np.zeros((masks.shape[0],masks.shape[1],len(mask_idxs)))

    models = []
    cids = np.zeros(0)
    scores = np.zeros(0, dtype='float32')
    model_transforms = []

    #Decompact into strictly binary mask stack
    for i,mask_idx in enumerate(mask_idxs):
        output[:,:,i] = np.where(masks == mask_idx,1,0)


        models.append(None)
        model_transforms.append(None)
        cids = np.append(cids, np.asarray(1))
        scores = np.append(scores, np.asarray(1.0, dtype='float32'))

    return (output, models, scores, cids, model_transforms)

def masks_from_Cellpose(mask_path=None, output_path=None, global_image_size=None,image_dir=None):
    '''
    Locates the cellpose export .npy files in the mask folder and generates single cell masks in the output folder, in the structure
    output_folder/annots/(image_name)/Cell1.bmp... Cell2.bmp....

    Parameters
    ----------
    mask_path : string
        path to directory with annotations
    output_path : string
        path to directory where the output data structure will be created.
        Function creates a folder called annots. Inside annots, each subdir is a separate image, inside which are binary masks.
    Returns
    -------

        '''

    print('Reading masks from Cellpose format.')
    print()

    used_masks = 0
    image_total = 0
    makedir(os.path.join(output_path, 'annots'))  # Create folder with masks


    # Find all annotation files that end with .json

    for root, dirs, files in os.walk(mask_path, topdown=True):
        for file in files:
            if file.endswith('.npy'):

                #Find all masks in file

                img_filename = os.path.splitext(file)[0]

                # Create folder with masks
                makedir(os.path.join(output_path, 'annots', img_filename))

                #Load annot file for image
                loadpath = os.path.join(root,file)

                masks, models, scores, class_ids, transforms = cellpose_file_loader(loadpath, image_size=None, completeload=False)

                #Save each mask

                maskcount = masks.shape[-1]
                cellcount = 0
                for i in range(maskcount):
                    mask = masks[:,:,i]

                    filename = 'Cell' + str(cellcount) + '.bmp'  # Mask filename
                    savepath = os.path.join(output_path, 'annots', img_filename,
                                            filename)  # Assemble whole save path

                    skimage.io.imsave(savepath, skimage.img_as_ubyte(mask), check_contrast=False)

                    cellcount += 1
                    used_masks += 1

                image_total += 1


    print('Generated', ':', str(used_masks), 'masks out of', str(image_total), 'images.')
    print('Generated {} masks from segmentation instances.'.format(used_masks))
    print()

    return os.path.join(output_path, 'annots')


def masks_from_integer_encode(mask_path=None, output_path=None, global_image_size=None, image_dir=None):
    '''
    Locates composite masks stored as .tif files (integer encoded) and generates single cell masks in the output folder,
    in the structure output_folder/annots/(image_name)/Cell1.bmp... Cell2.bmp....

        Parameters
    ----------
    mask_path : string
        path to directory with annotations
    output_path : string
        path to directory where the output data structure will be created.
        Function creates a folder called annots. Inside annots, each subdir is a separate image, inside which are binary masks.
    Returns
    -------

    '''

    print('Reading integer encoded masks')
    print()

    used_masks = 0
    image_total = 0

    # Find all annotation files that end with .tif

    for root, dirs, files in os.walk(mask_path, topdown=True):
        for file in files:
            if file.endswith('.tif'):

                #Find all masks in file

                img_filename = os.path.splitext(file)[0]

                # Create folder with masks
                makedir(os.path.join(output_path, 'annots', img_filename))

                #Load annot file for image
                loadpath = os.path.join(root,file)

                masks = skimage.io.imread(loadpath)

                #Extract individual cells
                mask_idxs = np.unique(masks)

                cellcount = 0

                # Remove degenerate case - the zero
                if 0 in mask_idxs:
                    i = np.where(mask_idxs == 0)
                    mask_idxs = np.delete(mask_idxs, i)

                output = np.zeros((masks.shape[0], masks.shape[1], len(mask_idxs)))

                # Decompact into strictly binary mask stack
                for mask_idx in mask_idxs:
                    singe_cell_mask = np.where(masks == mask_idx, 1, 0)

                    filename = 'Cell' + str(cellcount) + '.bmp'  # Mask filename
                    savepath = os.path.join(output_path, 'annots', img_filename,
                                            filename)  # Assemble whole save path

                    skimage.io.imsave(savepath, skimage.img_as_ubyte(single_cell_mask, check_contrast=False))

                    cellcount += 1
                    used_masks += 1

                image_total += 1

    print('Generated', ':', str(used_masks), 'masks out of', str(image_total), 'images.')
    print('Generated {} masks from segmentation instances.'.format(used_masks))
    print()

    return os.path.join(output_path, 'annots')



def masks_from_OUFTI(**kwargs):

    '''
    Locates the OUFTI export .mat files in the mask folder and generates single cell masks in the output folder, in the structure
    output_folder/annots/(image_name)/Cell1.png... Cell2.png....

    Parameters
    ----------
    mask_path : string
        path to directory with annotations
    output_path : string
        path to directory where the output data structure will be created.
        Function creates a folder called annots. Inside annots, each subdir is a separate image, inside which are binary masks.
    img_size : 2-tuple
        (x,y) image size, since metadata is not saved by OUFTI
    Returns
    -------

    '''

    mask_path = kwargs.get('mask_path', False)
    output_path = kwargs.get('output_path', False)
    image_size = kwargs.get('image_size', False)

    if not all([mask_path, output_path]):  # Verify input
        raise TypeError
    import scipy.io, skimage.draw, os, numpy, sys
    from PIL import Image

    makedir(os.path.join(output_path, 'annots'))  # Create folder with masks
    image_total = 0
    mask_total = 0
    error_count = 0
    meshless_cell_count = 0

    #Find all annotation files that end with .mat
    tracker = 0
    for root, dirs, files in os.walk(mask_path, topdown=True):
        for file in files:
            if file.endswith('.mat'):

                img_filename = file.split('.')[0]

                #Load annot file for image
                loadpath = os.path.join(root,file)
                annotfile = scipy.io.loadmat(loadpath)

                try:
                    cellData = annotfile['cellList']['meshData'][0][0][0][0][0] #Load data for all cells in image
                except:
                    error_count = error_count + 1
                    continue

                makedir(os.path.join(output_path, 'annots', img_filename))  # Create folder with masks

                cellcount=1
                for key,cell in dict(numpy.ndenumerate(cellData)).items(): #Iterate through all cells in image

                    mesh = cell['mesh'] #Get attribute

                    #Attempt to unpack. Skip if empty.
                    mesh = mesh[0][0]
                    if mesh.shape == (1,1):
                        meshless_cell_count = meshless_cell_count +1
                        continue

                    mesh = numpy.concatenate((mesh[:,:2],mesh[:,2:]),axis=0) #Reslice array to a more conventional format
                    mesh_tran = numpy.zeros(mesh.shape)

                    x=mesh[:,0]
                    y=mesh[:,1]

                    #21_01_2021 edit - subtract (1,1) from each vertex to account for difference in indexing between python and matlab

                    x = x-1
                    y = y-1

                    mesh_tran[:,0] = y #Swap columns to match polygon2mask
                    mesh_tran[:,1] = x

                    mask = skimage.draw.polygon2mask(image_size, mesh_tran)

                    if mask.max() == 0: #if no mask was written, skip
                        continue

                    filename = 'Cell' + str(cellcount) + '.bmp'  # Mask filename
                    savepath = os.path.join(output_path, 'annots', img_filename, filename)  # Assemble whole save path

                    Image.fromarray(mask).save(savepath)

                    mask_total = mask_total + 1
                    cellcount = cellcount + 1
                image_total = image_total + 1

    sys.stdout.flush()
    print('')
    print('Generated', ':', str(mask_total), 'masks out of', str(image_total), 'images.')
    print('cellList read errors:', error_count)
    print('Meshless cells found:', meshless_cell_count )
    sys.stdout.flush()

    return os.path.join(output_path, 'annots')


