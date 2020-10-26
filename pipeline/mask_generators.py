# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:50:28 2020

@author: Aleksander Zagajewski
"""

from helpers import *
import os
from tqdm import tqdm

def masks_from_VOTT(mask_path, output_path):
    '''
    Locates the VOTT json file in the mask folder and generates single cell masks in the output folder, in the structure
    output_folder/annots/(image_name)/Cell1.png... Cell2.png....

    Parameters
    ----------
    mask_path : string
        path to directory with annotations
    output_path : string
        p ath to directory where the output data structure will be created

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
    




if __name__ =='__main__':
    masks_from_VOTT(r'C:\Users\User\Desktop\WT+ETOH', r'C:\Users\User\Desktop\WT+ETOH\test')