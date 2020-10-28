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
    print('\n')
    print('')
    print('-------------------------------------------')
    print('Generated', ':', str(mask_total), 'masks out of', str(image_total), 'images.')
    print('-------------------------------------------')
    sys.stdout.flush()


if __name__ == '__main__':
    import os

    input_path = os.path.join(get_parent_path(1),'Data','Phenotype detection_18_08_20','Segregated','Combined','WT+ETOH','Segmentations')
    output_path = os.path.join(get_parent_path(1), 'Data', 'Phenotype detection_18_08_20', 'Segregated', 'Combined', 'WT+ETOH', 'Segmentations')
    masks_from_VOTT(input_path, output_path)
