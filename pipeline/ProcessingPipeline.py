# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:17:56 2020

@author: Aleksander Zagajewski
"""

from helpers import *
from implementations import *
from mask_generators import *
from segmentation import *

import sys

#sys.path.append(r"C:\Users\User\PycharmProjects\AMR\pipeline") #Append paths such that sub-processes can find functions
#sys.path.append(r"C:\Users\User\PycharmProjects\AMR\pipeline\helpers.py")
#sys.path.append(r"C:\Users\User\PycharmProjects\AMR\pipeline\implementations.py")



 
class ObjectFactory:
    
    def __init__(self):
        self._processes = {}
        
    def register_implementation(self, key, process):
        self._processes[key] = process
        
    def _create(self, key):
        try:
            process = self._processes[key]
            return process
        except KeyError:
            print(key, ' is not registered to an process.' )
            raise ValueError(key) 


class ProcessingPipeline:
    
    def __init__(self,path,instrument):
                        
        self._Factory = ObjectFactory()
        
        self.instrument = instrument
        self.path = path 
        self.opchain = []
        self.sorted = False
        self.collected = False

        self.segmenter = None


        #Use a 2 tuple as a key.
        self._Factory.register_implementation(('sorter','NIM'), SortNIM)
        self._Factory.register_implementation(('collector','NIM'), CollectNIM)

        self._Factory.register_implementation(('fileoperation','TrainTestVal_split'), TrainTestVal_split)
        self._Factory.register_implementation(('fileoperation', 'masks_from_VOTT'), masks_from_VOTT)
        self._Factory.register_implementation(('fileoperation', 'masks_from_OUFTI'), masks_from_OUFTI)
        self._Factory.register_implementation(('fileoperation', 'Equalize_Channels'), Equalize_Channels)

        self._Factory.register_implementation(('operation','BatchProcessor'), BatchProcessor)
        self._Factory.register_implementation(('operation', 'Imadjust'), Imadjust)
        self._Factory.register_implementation(('operation', 'Iminvert'), Iminvert)

        
    def Sort(self, **kwargs):
        instrument = self.instrument        
        sorter = self._Factory._create(('sorter',instrument)) #Fetch right sorter

        print('-------------------------')
        print('Executing Sort:', str(instrument))
        print('-------------------------')

        self.path = sorter(self.path,**kwargs) #call sorter and update path
        self.sorted = True #Set status flag
            
    def Collect(self, **kwargs):
        #assert self.sorted == True, 'Images must be sorted first.'
        
        instrument = self.instrument        
        collector = self._Factory._create(('collector',instrument)) #Fetch right collector

        print('-------------------------')
        print('Executing Collect:',str(instrument))
        print('-------------------------')

        self.path = collector(self.path,**kwargs) #call and and update path
        self.collected = True #Set status flag

    def FileOp(self, op, **kwargs):

        operation = self._Factory._create(('fileoperation',op))

        print('-------------------------')
        print('Executing:',str(op))
        print('-------------------------')

        operation(**kwargs)

        self.opchain.append(str(operation))

    def ImageOp(self, op, **kwargs):
            
        batch_processor = self._Factory._create(('operation','BatchProcessor'))
        operation = self._Factory._create(('operation', op))

        print('-------------------------')
        print('Executing:',str(op))
        print('-------------------------')

        batch_processor(self.path, operation, op, **kwargs )

        self.opchain.append(str(operation))


    
    
if __name__ == '__main__':

    import os
    data_folder = os.path.join(get_parent_path(1),'Data','Phenotype detection_18_08_20')
    
    cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
    image_channels = ['NR','NR','NR']
    img_dims = (684,840,30)
    
    pipeline = ProcessingPipeline(data_folder, 'NIM')
    pipeline.Sort(cond_IDs = cond_IDs, dims = img_dims, image_channels = image_channels)
    pipeline.Collect(cond_IDs = cond_IDs, image_channels = image_channels)


    #--- GENERATE MASKS FROM SEGMENTATION FILE---

    input_path_WT = os.path.join(get_parent_path(1), 'Data','Phenotype detection_18_08_20', 'Segmentations', 'WT+ETOH')
    input_path_CIP = os.path.join(get_parent_path(1), 'Data','Phenotype detection_18_08_20', 'Segmentations', 'CIP+ETOH')
    input_path_RIF = os.path.join(get_parent_path(1), 'Data','Phenotype detection_18_08_20', 'Segmentations', 'RIF+ETOH')

    pipeline.FileOp('masks_from_OUFTI', mask_path=input_path_WT, output_path = input_path_WT, image_size = (684,420))
    pipeline.FileOp('masks_from_OUFTI', mask_path=input_path_CIP, output_path= input_path_CIP, image_size=(684, 420))
    pipeline.FileOp('masks_from_OUFTI', mask_path=input_path_RIF, output_path= input_path_RIF, image_size=(684, 420))

    #--- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
    annots_WT = os.path.join(input_path_WT, 'annots')
    files_WT = os.path.join(get_parent_path(1),'Data','Phenotype detection_18_08_20', 'Segregated', 'Combined', 'WT+ETOH')

    annots_CIP = os.path.join(input_path_CIP, 'annots')
    files_CIP = os.path.join(get_parent_path(1), 'Data', 'Phenotype detection_18_08_20', 'Segregated', 'Combined', 'CIP+ETOH')

    annots_RIF = os.path.join(input_path_RIF, 'annots')
    files_RIF = os.path.join(get_parent_path(1), 'Data', 'Phenotype detection_18_08_20', 'Segregated', 'Combined','RIF+ETOH')

    output = os.path.join(get_parent_path(1),'Data', 'Dataset1_15_01_2021')

    pipeline.FileOp('TrainTestVal_split', data_sources = [files_WT,files_CIP,files_RIF], annotation_sources = [annots_WT,annots_CIP,annots_RIF], output_folder = output, proportions = (0.7,0.2,0.1), seed = 40 )

    #---TRAIN 1ST STAGE MODEL---

    weights_start = os.path.join(get_parent_path(1), 'Data','mask_rcnn_coco.h5')
    train_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset1_15_01_2021', 'Train')
    val_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset1_15_01_2021', 'Validation')
    output_dir = get_parent_path(1)


    configuration = BacConfig()

    train_mrcnn_segmenter(train_folder = train_dir, validation_folder = val_dir, configuration = configuration, weights = weights_start, output_folder = output_dir)





    
