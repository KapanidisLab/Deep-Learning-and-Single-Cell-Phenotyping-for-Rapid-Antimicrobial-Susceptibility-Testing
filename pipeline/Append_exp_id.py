import os

from helpers import *


EXP_ID = 4

dirpath = os.path.join(get_parent_path(1), 'Data', 'Phenotype_detection_repeats/03_04_21/Segmentations_All')

for root,dir,files in os.walk(dirpath):
    for file in files:

        if file.endswith('.tif') or file.endswith('.mat'):

            if file.startswith(str(EXP_ID)):
                continue

            filepath = os.path.join(root,file)

            new_file = str(EXP_ID) + file

            newfilepath = os.path.join(root,new_file)

            os.rename(filepath,newfilepath)

