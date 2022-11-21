# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:17:56 2020

@author: Aleksander Zagajewski
"""

from helpers import *
from implementations import *
from mask_generators import *
from segmentation import *
from classification import *

import numpy as np, os

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
        self._Factory.register_implementation(('sorter','NIM'), SortNIM2)
        self._Factory.register_implementation(('sorter','NIM-CF'), SortNIM2_CF)

        self._Factory.register_implementation(('collector','NIM'), CollectNIM2)
        self._Factory.register_implementation(('collector', 'NIM-CF'), CollectNIM2)

        self._Factory.register_implementation(('fileoperation','TrainTestVal_split'), TrainTestVal_split)
        self._Factory.register_implementation(('fileoperation', 'masks_from_VOTT'), masks_from_VOTT)
        self._Factory.register_implementation(('fileoperation', 'masks_from_OUFTI'), masks_from_OUFTI)
        self._Factory.register_implementation(('fileoperation', 'masks_from_Cellpose'), masks_from_Cellpose)
        self._Factory.register_implementation(('fileoperation', 'masks_from_integer_encoding'), masks_from_integer_encoding)
        self._Factory.register_implementation(('fileoperation', 'masks_from_integer_encoding_CF'), masks_from_integer_encoding_CF)

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

        self.path, stats = collector(self.path,**kwargs) #call and and update path
        self.collected = True #Set status flag
        return stats

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


    
