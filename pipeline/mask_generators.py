# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:50:28 2020

@author: Aleksander Zagajewski
"""

from helpers import *
import os
from tqdm import tqdm

def masks_from_VOTT(mask_path, output_path):

    import json, numpy
    '''
    Locates the VOTT json file in the mask folder and generates single cell masks in the output folder, in the structure
    output_folder/annots/(image_name)/Cell1.png... Cell2.png....

    Parameters
    ----------
    mask_path : string
        path to directory with annotations
    output_path : string
        path to directory where the output data structure will be created

    Returns
    -------

    '''
    tracker = 0
    for root, dirs, files in tqdm(os.walk(mask_path, topdown=True), total = dircounter(mask_path), unit = 'files', desc = 'Searching directory for export file.'):
        for file in files:
            if file.endswith('export.json'):
                tracker = tracker + 1
                export_file = os.path.join(root, file)

    assert tracker == 1, 'Warning - either no export file found, or multiple copies exist in given folder. Aborting.'

    with open(export_file) as json_file: #Navigate through json structure and extract information
        data = json.load(json_file)
        assets = data['assets']

        for image_key in assets:
            image = assets[image_key]
            image_metadata = image['asset']
            image_filename, image_size = image_metadata['name'], image_metadata['size']




            for region_key in image:









if __name__ == '__main__':
    import os

    input_path = os.path.join(get_parent_path(1),'Data','Phenotype detection_18_08_20','Segregated','Combined','WT+ETOH','Segmentations')
    data = masks_from_VOTT(input_path, 'asdads')
