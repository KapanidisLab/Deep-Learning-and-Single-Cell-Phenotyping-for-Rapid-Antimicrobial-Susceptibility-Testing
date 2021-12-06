from ProcessingPipeline import ProcessingPipeline
import os
from helpers import *

from classification import *

# ---PREPARE DATASET WITH BOTH CHANNELS

data_folder = os.path.join(get_parent_path(1),'Data','Exp1')
output_segregated = os.path.join(get_parent_path(1),'Data','Exp1_Segregated_Multichannel')

output_collected = os.path.join(get_parent_path(1),'Data','Exp1_Collected_Multichannel')

cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
image_channels = ['NR','DAPI']
img_dims = (30,684,840)


pipeline = ProcessingPipeline(data_folder, 'NIM')
pipeline.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0, 'NR':0}, output_folder=output_segregated)
pipeline.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected, registration_target=0)

# ---GENERATE CELLS DATASET FROM SEGMENTATION MASKS AND BOTH CHANNELS, SPLIT AND SAVE. TRAIN AND TEST SEPARATELY

input_path_WT = os.path.join(get_parent_path(1), 'Data', 'Segmentations_All', 'WT+ETOH')
input_path_CIP = os.path.join(get_parent_path(1), 'Data', 'Segmentations_All', 'CIP+ETOH')
input_path_RIF = os.path.join(get_parent_path(1), 'Data', 'Segmentations_All', 'RIF+ETOH')

#pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_WT, output_path = input_path_WT)
#pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_CIP, output_path= input_path_CIP)
#pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_RIF, output_path= input_path_RIF)

#--- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
annots_WT = os.path.join(input_path_WT, 'annots')
files_WT = os.path.join(output_collected,'WT+ETOH')

annots_CIP = os.path.join(input_path_CIP, 'annots')
files_CIP = os.path.join(output_collected,'CIP+ETOH')

annots_RIF = os.path.join(input_path_RIF, 'annots')
files_RIF = os.path.join(output_collected,'RIF+ETOH')

output = os.path.join(get_parent_path(1),'Data', 'Dataset_Exp1_Multichannel')

#pipeline.FileOp('TrainTestVal_split', data_sources = [files_WT,files_CIP,files_RIF], annotation_sources = [annots_WT,annots_CIP,annots_RIF], output_folder=output,test_size = 0.2, validation_size=0.2, seed=42 )

print('extracting manual struct')
manual_struct = struct_from_file(dataset_folder=os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp1_Multichannel'),
                                 class_id=1)

cells = cells_from_struct(input=manual_struct, cond_IDs=cond_IDs, image_dir=output_collected, mode='masks')
X_train, X_test, y_train, y_test = split_cell_sets(input=cells, test_size=0.2, random_state=42)

#Fix pycharm console
class PseudoTTY(object):
    def __init__(self, underlying):
        self.__underlying = underlying
    def __getattr__(self, name):
        return getattr(self.__underlying, name)
    def isatty(self):
        return True

sys.stdout = PseudoTTY(sys.stdout)

# ---TRAIN---
logdir = os.path.join(get_parent_path(1), 'Second_Stage_2')
resize_target = (64, 64, 3)
class_count = 3

#train(mode='DenseNet121_test', X_train=X_train, y_train=y_train, size_target=resize_target, class_count=class_count, pad_cells=True,
 #     logdir=logdir, batch_size=16, epochs=100,learning_rate=0.0005, optimizer='NAdam', verbose=True, dt_string='DenseNet121')
#print('inspecting')
#inspect(modelpath=os.path.join(logdir,'DenseNet121.h5'), X_test=X_test, y_test=y_test, mean=np.asarray([0, 0, 0]), size_target=resize_target, pad_cells=True,
 #       class_id_to_name=cells['class_id_to_name'])
