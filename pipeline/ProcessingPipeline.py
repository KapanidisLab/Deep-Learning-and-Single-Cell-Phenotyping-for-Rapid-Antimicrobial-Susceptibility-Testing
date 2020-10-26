# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:17:56 2020

@author: Aleksander Zagajewski
"""

from helpers import *
from implementations import * 

import sys

sys.path.append(r"C:\Users\User\PycharmProjects\AMR\pipeline") #Append paths such that sub-processes can find functions
sys.path.append(r"C:\Users\User\PycharmProjects\AMR\pipeline\helpers.py")
sys.path.append(r"C:\Users\User\PycharmProjects\AMR\pipeline\implementations.py")



 
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
            
    def ImageOp(self, op, **kwargs):
            
        batch_processor = self._Factory._create(('operation','BatchProcessor'))
        operation = self._Factory._create(('operation', op))
            
        batch_processor(self.path, operation, op, **kwargs )
            
        
        self.opchain.append(str(operation))

            
    
    
if __name__ == '__main__':
        
    data_folder = r'C:\Users\zagajewski\Desktop\Phenotype detection_18_08_20'
    
    cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
    image_channels = ['NR']
    img_dims = (684,840,30)
    
    pipeline = ProcessingPipeline(data_folder, 'NIM')
    #pipeline.Sort(path = data_folder,cond_IDs = cond_IDs, dims = img_dims, image_channels = image_channels)
    pipeline.Collect(path = data_folder,cond_IDs = cond_IDs, image_channels = image_channels)
    
   # pipeline.ImageOp('Imadjust', index = 0)
   # pipeline.ImageOp('WaveletEnhance', sigmas = (1,5), mu = 2, scales = (1,200,1), index = 0)
   # pipeline.ImageOp('Iminvert', index = 0)
    
