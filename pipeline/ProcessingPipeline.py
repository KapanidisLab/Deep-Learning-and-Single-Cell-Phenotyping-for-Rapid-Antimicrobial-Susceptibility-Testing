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
        self._Factory.register_implementation(('collector','NIM'), CollectNIM2)

        self._Factory.register_implementation(('fileoperation','TrainTestVal_split'), TrainTestVal_split)
        self._Factory.register_implementation(('fileoperation', 'masks_from_VOTT'), masks_from_VOTT)
        self._Factory.register_implementation(('fileoperation', 'masks_from_OUFTI'), masks_from_OUFTI)
        self._Factory.register_implementation(('fileoperation', 'masks_from_Cellpose'), masks_from_Cellpose)
        self._Factory.register_implementation(('fileoperation', 'masks_from_integer_encoding'), masks_from_integer_encoding)
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
        assert self.sorted == True, 'Images must be sorted first.'
        
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


    
    
if __name__ == '__main__':

    import os

    data_folder_train = os.path.join(get_parent_path(1),'Data','Train_0+3')
    output_segregated_train = os.path.join(get_parent_path(1),'Data','Train_0+3_singlechannel_segregated')
    output_collected_train = os.path.join(get_parent_path(1),'Data','Train_0+3_singlechannel_collected')

    data_folder_test = os.path.join(get_parent_path(1),'Data','Test_4')
    output_segregated_test = os.path.join(get_parent_path(1),'Data','Test_4_singlechannel_segregated')
    output_collected_test = os.path.join(get_parent_path(1),'Data','Test_4_singlechannel_collected')


    cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
    image_channels = ['NR','NR','NR']
    img_dims = (30,684,840)

    pipeline = ProcessingPipeline(data_folder_train, 'NIM')
    pipeline.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0,'NR':0}, output_folder=output_segregated_train)
    pipeline.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected_train, registration_target=None)

    pipeline2 = ProcessingPipeline(data_folder_test,'NIM')
    pipeline2.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0,'NR':0}, output_folder=output_segregated_test)
    pipeline2.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected_test, registration_target=None)
    #--- GENERATE MASKS FROM SEGMENTATION FILE---

    input_path_WT = os.path.join(get_parent_path(1), 'Data','Train_0+3', 'Segmentations', 'WT+ETOH')
    input_path_CIP = os.path.join(get_parent_path(1), 'Data','Train_0+3', 'Segmentations', 'CIP+ETOH')
    input_path_RIF = os.path.join(get_parent_path(1), 'Data','Train_0+3', 'Segmentations', 'RIF+ETOH')

    pipeline.FileOp('masks_from_OUFTI', mask_path=input_path_WT, output_path = input_path_WT, image_size=(684, 420))
    pipeline.FileOp('masks_from_OUFTI', mask_path=input_path_CIP, output_path= input_path_CIP, image_size=(684, 420))
    pipeline.FileOp('masks_from_OUFTI', mask_path=input_path_RIF, output_path= input_path_RIF, image_size=(684, 420))



    #--- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
    annots_WT = os.path.join(input_path_WT, 'annots')
    files_WT = os.path.join(output_collected_train,'WT+ETOH')

    annots_CIP = os.path.join(input_path_CIP, 'annots')
    files_CIP = os.path.join(output_collected_train,'CIP+ETOH')

    annots_RIF = os.path.join(input_path_RIF, 'annots')
    files_RIF = os.path.join(output_collected_train,'RIF+ETOH')

    output = os.path.join(get_parent_path(1),'Data', 'Dataset_Train0+3')

    pipeline.FileOp('TrainTestVal_split', data_sources = [files_WT,files_CIP,files_RIF], annotation_sources = [annots_WT,annots_CIP,annots_RIF], output_folder = output,test_size = 0, validation_size=0.2, seed = 42 )

    input_path_WT = os.path.join(get_parent_path(1), 'Data', 'Test_4', 'Segmentations', 'WT+ETOH')
    input_path_CIP = os.path.join(get_parent_path(1), 'Data', 'Test_4', 'Segmentations', 'CIP+ETOH')
    input_path_RIF = os.path.join(get_parent_path(1), 'Data', 'Test_4', 'Segmentations', 'RIF+ETOH')

    pipeline2.FileOp('masks_from_OUFTI', mask_path=input_path_WT, output_path=input_path_WT, image_size=(684, 420))
    pipeline2.FileOp('masks_from_OUFTI', mask_path=input_path_CIP, output_path=input_path_CIP, image_size=(684, 420))
    pipeline2.FileOp('masks_from_OUFTI', mask_path=input_path_RIF, output_path=input_path_RIF, image_size=(684, 420))

    # --- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
    annots_WT = os.path.join(input_path_WT, 'annots')
    files_WT = os.path.join(output_collected_test, 'WT+ETOH')

    annots_CIP = os.path.join(input_path_CIP, 'annots')
    files_CIP = os.path.join(output_collected_test, 'CIP+ETOH')

    annots_RIF = os.path.join(input_path_RIF, 'annots')
    files_RIF = os.path.join(output_collected_test, 'RIF+ETOH')

    output = os.path.join(get_parent_path(1), 'Data', 'Dataset_Test4')

    pipeline.FileOp('TrainTestVal_split', data_sources=[files_WT, files_CIP, files_RIF],
                    annotation_sources=[annots_WT, annots_CIP, annots_RIF], output_folder=output, test_size=1.0,
                    validation_size=0, seed=42)
    
    #---TRAIN 1ST STAGE MODEL---

    weights_start = os.path.join(get_parent_path(1), 'Data','mask_rcnn_coco.h5')
    train_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset_Train0+3', 'Train')
    val_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset_Train0+3', 'Validation')
    test_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset_Test4', 'Test')
    output_dir = get_parent_path(1)

    configuration = BacConfig()
    configuration.NAME = 'PredConfig_1+3'

    import imgaug.augmenters as iaa  # import augmentation library

    augmentation = [
        iaa.Fliplr(0.5),  # Flip LR with 50% probability
        iaa.Flipud(0.5),  # Flip UD 50% prob
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45))),  # Rotate up to 45 deg either way, 50% prob
        iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})),
        # Translate up to 20% on either axis independently, 50% prob
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),  # Gaussian convolve 50% prob
        # iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 65535))),  # up to 5% PSNR 50% prob
        iaa.Sometimes(0.5, iaa.Cutout(nb_iterations=(1, 10), size=0.05, squared=False, cval=0))
    ]

    # --- INSPECT TRAIN DATASET AND AUGMENTATION---

    inspect_dataset(dataset_folder = train_dir)
    inspect_augmentation(dataset_folder = train_dir, configuration = configuration, augmentation = augmentation)

    # --- TRAIN 1st STAGE SEGMENTER

    #train_mrcnn_segmenter(train_folder = train_dir, validation_folder = val_dir, configuration = configuration, augmentation = augmentation, weights = weights_start, output_folder = output_dir)

    # --- INSPECT 1st STAGE STEPWISE AND OPTIMISE

    evaluate_coco_metrics(dataset_folder=test_dir,config=config,weights = os.path.join(get_parent_path(1), "predconfig_1+320211013T2353","mask_rcnn_predconfig_1+3.h5"))

    #inspect_segmenter_stepwise(train_folder = train_dir, test_folder = test_dir, configuration = configuration, weights = weights)
    #optimise_mrcnn_segmenter(mode = 'training', arg_names = ['LEARNING_RATE', 'IMAGES_PER_GPU'], arg_values = [[0.007,0.01],[4,6,8]], train_folder = train_dir, validation_folder = val_dir, configuration = configuration, augmentation = augmentation, weights = weights_start, output_folder = output_dir )
    #optimise_mrcnn_segmenter(mode = 'inference', arg_names = ['DETECTION_NMS_THRESHOLD' ], arg_values = [[0.2,0.1]], test_folder=test_dir, configuration=configuration, weights=weights, ids=ids)
    #inspect_mrcnn_segmenter(test_folder = test_dir, configuration = configuration, weights = weights, ids=ids )

#---------------------------------------------------------------------------------------------------------------------------------

    '''
    

    #weights = os.path.join(get_parent_path(1), "predconfig_1+320211013T2353", "mask_rcnn_predconfig_1+3.h5")

    #output_struct = predict_mrcnn_segmenter(source = test_dir, mode = 'dataset', config = configuration, weights = weights)
    
    #---PREPARE DATASET WITH BOTH CHANNELS

    multichannel_folder = os.path.join(get_parent_path(1), data_folder, 'multichannel')
    image_channels = ['NR','DAPI']

    pipeline_cells = ProcessingPipeline(data_folder, 'NIM')
    pipeline_cells.Sort(cond_IDs=cond_IDs, dims=img_dims, image_channels=image_channels, output_folder = multichannel_folder)
    pipeline_cells.Collect(cond_IDs=cond_IDs, image_channels=image_channels, registration_target=0)

    #---GENERATE CELLS DATASET FROM SEGMENTATION MASKS AND BOTH CHANNELS, SPLIT AND SAVE

    manual_struct = struct_from_file(dataset_folder=os.path.join(get_parent_path(1), 'Data', 'Dataset_Train1+3_14_08_2021'),
                                     class_id=1)

    cells = cells_from_struct(input=manual_struct, cond_IDs=cond_IDs, image_dir=pipeline_cells.path, mode='masks')

    X_train, X_test, y_train, y_test = split_cell_sets(input=cells, test_size=0.2, random_state=42)

    #---TRAIN---
    logdir = os.path.join(get_parent_path(1),'Second_Stage_2')
    resize_target = (64,64,3)
    class_count = 3

    train(mode='ResNet50', X_train=X_train, y_train=y_train, resize_target=resize_target, class_count=class_count, logdir=None, batch_size=64)

    #---PREDICT---
    modelname = 'ResNet50_BS_64_LR_0001_opt_NAdam'
    path = os.path.join(logdir, modelname+'.h5' )
    mean = np.asarray([0,0,0])

    inspect(modelpath=path, X_test=X_test, y_test=y_test, mean=mean, resize_target=resize_target,class_id_to_name=cells['class_id_to_name'])
    
    '''