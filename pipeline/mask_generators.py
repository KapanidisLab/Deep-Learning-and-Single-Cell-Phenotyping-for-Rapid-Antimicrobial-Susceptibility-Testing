# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:50:28 2020

@author: Aleksander Zagajewski
"""

from helpers import *
import os
from tqdm import tqdm

def masks_from_VOTT(**kwargs):

    import json, numpy, skimage,skimage.io,skimage.draw, sys

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
    import scipy.io, skimage.io, skimage.draw, os, numpy, sys
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
                    mesh_tran[:,0] = y #Swap columns to match polygon2mask
                    mesh_tran[:,1] = x

                    mask = skimage.draw.polygon2mask(image_size, mesh_tran)

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



if __name__ == '__main__':
    import os

    input_path = os.path.join(r'C:\Users\zagajewski\Desktop\AMR_png_images_for_segmentation\18_08_20\WT_fakePC')
    output_path = os.path.join(r'C:\Users\zagajewski\Desktop\AMR_png_images_for_segmentation\18_08_20\WT_fakePC')
    masks_from_OUFTI(mask_path=input_path, output_path=output_path)