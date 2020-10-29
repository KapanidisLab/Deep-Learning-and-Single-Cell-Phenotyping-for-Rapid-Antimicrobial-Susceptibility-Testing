# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:17:56 2020

@author: Aleksander Zagajewski
"""

from helpers import *
from implementations import *
from mask_generators import *

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

        #Use a 2 tuple as a key.
        self._Factory.register_implementation(('sorter','NIM'), SortNIM)
        self._Factory.register_implementation(('collector','NIM'), CollectNIM)

        self._Factory.register_implementation(('fileoperation','TrainTestVal_split'), TrainTestVal_split)
        self._Factory.register_implementation(('fileoperation', 'masks_from_VOTT'), masks_from_VOTT)

        self._Factory.register_implementation(('operation','BatchProcessor'), BatchProcessor)
        self._Factory.register_implementation(('operation', 'Imadjust'), Imadjust)
        self._Factory.register_implementation(('operation', 'Iminvert'), Iminvert)

        
    def Sort(self, **kwargs):
        instrument = self.instrument        
        sorter = self._Factory._create(('sorter',instrument)) #Fetch right sorter

        self.path = sorter(self.path,**kwargs) #call sorter and update path       
        
        self.sorted = True #Set status flag
            
    def Collect(self, **kwargs):
        #assert self.sorted == True, 'Images must be sorted first.'
        
        instrument = self.instrument        
        collector = self._Factory._create(('collector',instrument)) #Fetch right collector
        self.path = collector(self.path,**kwargs) #call and and update path
        
        self.collected = True #Set status flag

    def FileOp(self, op, **kwargs):

        operation = self._Factory._create(('fileoperation',op))
        operation(**kwargs)

        self.opchain.append(str(operation))

    def ImageOp(self, op, **kwargs):
            
        batch_processor = self._Factory._create(('operation','BatchProcessor'))
        operation = self._Factory._create(('operation', op))
            
        batch_processor(self.path, operation, op, **kwargs )
            
        
        self.opchain.append(str(operation))

            
    
    
if __name__ == '__main__':

    import os
    data_folder = os.path.join(get_parent_path(1),'Data','Phenotype detection_18_08_20')
    
    cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
    image_channels = ['NR','DAPI']
    img_dims = (684,840,30)
    
    pipeline = ProcessingPipeline(data_folder, 'NIM')
    pipeline.Sort(path = data_folder,cond_IDs = cond_IDs, dims = img_dims, image_channels = image_channels)
    pipeline.Collect(path = data_folder,cond_IDs = cond_IDs, image_channels = image_channels)


    #--- GENERATE MASKS FROM SEGMENTATION FILE---

    input_path = os.path.join(get_parent_path(1), 'Data', 'Phenotype detection_18_08_20', 'Segregated', 'Combined',
                              'WT+ETOH', 'Segmentations')
    output_path = input_path #Write in same directory

    pipeline.FileOp('masks_from_VOTT', mask_path = input_path, output_path = output_path)

    #--- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS---
    annots = os.path.join(input_path, 'annots')
    files = os.path.join(get_parent_path(1),'Data','Phenotype detection_18_08_20', 'Segregated', 'Combined', 'WT+ETOH')
    output = os.path.join(get_parent_path(1),'Data', 'Dataset1_27_10_20')

    pipeline.FileOp('TrainTestVal_split', data_folder = files, annotation_folder = annots, output_folder = output, proportions = (0.7,0.2,0.1), seed = 42 )


   # pipeline.ImageOp('Imadjust', index = 0)
   # pipeline.ImageOp('WaveletEnhance', sigmas = (1,5), mu = 2, scales = (1,200,1), index = 0)
   # pipeline.ImageOp('Iminvert', index = 0)
    
