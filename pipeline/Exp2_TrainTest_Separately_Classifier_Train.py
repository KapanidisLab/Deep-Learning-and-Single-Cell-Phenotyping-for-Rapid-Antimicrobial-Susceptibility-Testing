from pipeline.ProcessingPipeline import ProcessingPipeline
import os
from pipeline.helpers import *

from pipeline.classification import *
import numpy as np

# weights = os.path.join(get_parent_path(1), "predconfig_1+320211013T2353", "mask_rcnn_predconfig_1+3.h5")

# ---PREPARE DATASET WITH BOTH CHANNELS

data_folder_train = os.path.join(get_parent_path(1),'Data','Exp2','Train')
output_segregated_train = os.path.join(get_parent_path(1),'Data','Exp2','Exp2_Segregated_Multichannel_Train')
output_collected_train = os.path.join(get_parent_path(1),'Data','Exp2','Exp2_Collected_Multichannel_Train')

data_folder_test = os.path.join(get_parent_path(1),'Data','Exp2','Test')
output_segregated_test = os.path.join(get_parent_path(1),'Data','Exp2','Exp2_Segregated_Multichannel_Test')
output_collected_test = os.path.join(get_parent_path(1),'Data','Exp2','Exp2_Collected_Multichannel_Test')


cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
image_channels = ['NR','DAPI']
img_dims = (30,684,840)

pipeline_train = ProcessingPipeline(data_folder_train, 'NIM')
'''
pipeline_train.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0, 'NR':0}, output_folder=output_segregated_train)
pipeline_train.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected_train, registration_target=0)
'''

pipeline_test = ProcessingPipeline(data_folder_test, 'NIM')
'''
pipeline_test.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0, 'NR':0}, output_folder=output_segregated_test)
pipeline_test.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected_test, registration_target=0)
'''
# ---GENERATE CELLS DATASETS FROM SEGMENTATION MASKS AND BOTH CHANNELS, SPLIT AND SAVE. TRAIN AND TEST SEPARATELY
'''
input_path_WT = os.path.join(get_parent_path(1), 'Data', 'Segmentations_200PerExperiment', 'WT+ETOH')
input_path_CIP = os.path.join(get_parent_path(1), 'Data', 'Segmentations_200PerExperiment', 'CIP+ETOH')
input_path_RIF = os.path.join(get_parent_path(1), 'Data', 'Segmentations_200PerExperiment', 'RIF+ETOH')


pipeline_train.FileOp('masks_from_integer_encoding', mask_path=input_path_WT, output_path = input_path_WT)
pipeline_train.FileOp('masks_from_integer_encoding', mask_path=input_path_CIP, output_path= input_path_CIP)
pipeline_train.FileOp('masks_from_integer_encoding', mask_path=input_path_RIF, output_path= input_path_RIF)


annots_WT = os.path.join(input_path_WT, 'annots')
annots_CIP = os.path.join(input_path_CIP, 'annots')
annots_RIF = os.path.join(input_path_RIF, 'annots')

output = os.path.join(get_parent_path(1),'Data', 'Dataset_Exp2_Multichannel_Train')

files_WT = os.path.join(output_collected_train,'WT+ETOH')
files_CIP = os.path.join(output_collected_train,'CIP+ETOH')
files_RIF = os.path.join(output_collected_train,'RIF+ETOH')
pipeline_train.FileOp('TrainTestVal_split', data_sources = [files_WT,files_CIP,files_RIF], annotation_sources = [annots_WT,annots_CIP,annots_RIF], output_folder=output,test_size = 0, validation_size=0.2, seed=42 )

output = os.path.join(get_parent_path(1),'Data', 'Dataset_Exp2_Multichannel_Test')

files_WT = os.path.join(output_collected_test,'WT+ETOH')
files_CIP = os.path.join(output_collected_test,'CIP+ETOH')
files_RIF = os.path.join(output_collected_test,'RIF+ETOH')
pipeline_test.FileOp('TrainTestVal_split', data_sources = [files_WT,files_CIP,files_RIF], annotation_sources = [annots_WT,annots_CIP,annots_RIF], output_folder=output,test_size = 1, validation_size=0, seed=42 )

'''



manual_struct_train = struct_from_file(dataset_folder=os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp2_Multichannel_Train'),
                                 class_id=1)
cells_train = cells_from_struct(input=manual_struct_train, cond_IDs=cond_IDs, image_dir=output_collected_train, mode='masks')
X_train, _, y_train, _ = split_cell_sets(input=cells_train, test_size=0, random_state=42)

manual_struct_test = struct_from_file(dataset_folder=os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp2_Multichannel_Test'),
                                 class_id=1)

cells_test = cells_from_struct(input=manual_struct_test, cond_IDs=cond_IDs, image_dir=output_collected_test, mode='masks')
_, X_test, _, y_test = split_cell_sets(input=cells_test, test_size=1, random_state=42)



# ---TRAIN---
logdir = os.path.join(get_parent_path(1), 'Second_Stage_2')
size_target = (64, 64, 3)
class_count = 3

#Fix pycharm console
class PseudoTTY(object):
    def __init__(self, underlying):
        self.__underlying = underlying
    def __getattr__(self, name):
        return getattr(self.__underlying, name)
    def isatty(self):
        return True

sys.stdout = PseudoTTY(sys.stdout)

train(mode='DenseNet121', X_train=X_train, y_train=y_train, size_target=size_target, pad_cells=True, class_count=class_count,
      logdir=logdir, batch_size=16, epochs=100,learning_rate=0.0005, optimizer='NAdam', verbose=True, dt_string='DenseNet121_EXP2_TOP200PerExp')

inspect(modelpath=os.path.join(logdir,'DenseNet121_EXP2_TOP200PerExp.h5'), X_test=X_test, y_test=y_test, mean=np.asarray([0, 0, 0]), size_target=size_target, pad_cells=True,
        class_id_to_name=cells_train['class_id_to_name'])