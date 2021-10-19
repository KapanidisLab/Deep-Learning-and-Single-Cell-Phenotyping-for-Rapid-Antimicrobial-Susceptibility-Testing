from ProcessingPipeline import ProcessingPipeline
import os
from helpers import *

from classification import *


# weights = os.path.join(get_parent_path(1), "predconfig_1+320211013T2353", "mask_rcnn_predconfig_1+3.h5")

# ---PREPARE DATASET WITH BOTH CHANNELS

data_folder_train = os.path.join(get_parent_path(1),'Data','Train_0+3')
output_segregated_train = os.path.join(get_parent_path(1),'Data','Train_0+3_multichannel_segregated')
output_collected_train = os.path.join(get_parent_path(1),'Data','Train_0+3_multichannel_collected')

data_folder_test = os.path.join(get_parent_path(1),'Data','Test_4')
output_segregated_test = os.path.join(get_parent_path(1),'Data','Test_4_multichannel_segregated')
output_collected_test = os.path.join(get_parent_path(1),'Data','Test_4_multichannel_collected')


cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
image_channels = ['NR','DAPI']
img_dims = (30,684,840)

pipeline_train = ProcessingPipeline(data_folder_train, 'NIM')
pipeline_train.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0, 'NR':0}, output_folder=output_segregated_train)
pipeline_train.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected_train, registration_target=0)

pipeline_test = ProcessingPipeline(data_folder_test, 'NIM')
pipeline_test.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0, 'NR':0}, output_folder=output_segregated_test)
pipeline_test.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected_test, registration_target=0)
# ---GENERATE CELLS DATASET FROM SEGMENTATION MASKS AND BOTH CHANNELS, SPLIT AND SAVE. TRAIN AND TEST SEPARATELY

manual_struct_train = struct_from_file(dataset_folder=os.path.join(get_parent_path(1), 'Data', 'Dataset_Train0+3'),
                                 class_id=1)
cells_train = cells_from_struct(input=manual_struct_train, cond_IDs=cond_IDs, image_dir=pipeline_train.path, mode='masks')
X_train, _, y_train, _ = split_cell_sets(input=cells_train, test_size=0, random_state=42)

manual_struct_test = struct_from_file(dataset_folder=os.path.join(get_parent_path(1), 'Data', 'Dataset_Test4'),
                                 class_id=1)
cells_test = cells_from_struct(input=manual_struct_test, cond_IDs=cond_IDs, image_dir=pipeline_test.path, mode='masks')
_, X_test, _, y_test = split_cell_sets(input=cells_test, test_size=1, random_state=42)


# ---TRAIN---
logdir = os.path.join(get_parent_path(1), 'Second_Stage_2')
resize_target = (64, 64, 3)
class_count = 3

parameter_grid = {'batch_size':[8,16,32,64], 'learning_rate':[0.0005,0.001,0.005],'epochs':[100],'optimizer':['SGD+N','SGD','NAdam','Adam']}

#optimize(mode = 'VGG16', X_train = X_train, y_train = y_train, parameter_grid = parameter_grid, resize_target = resize_target, class_count = class_count, logdir = logdir )
#optimize(mode = 'ResNet50', X_train = X_train, y_train = y_train, parameter_grid = parameter_grid, resize_target = resize_target, class_count = class_count, logdir = logdir )
#optimize(mode = 'DenseNet121', X_train = X_train, y_train = y_train, parameter_grid = parameter_grid, resize_target = resize_target, class_count = class_count, logdir = logdir )

train(mode='ResNet50', X_train=X_train, y_train=y_train, resize_target=resize_target, class_count=class_count,
      logdir=None, batch_size=64)


# ---PREDICT---
modelname = 'DenseNet121 BS 16, LR 0.0005, epochs 100, opt NAdam'
path = os.path.join(logdir,modelname, modelname + '.h5')
mean = np.asarray([0, 0, 0])

inspect(modelpath=path, X_test=X_test, y_test=y_test, mean=mean, resize_target=resize_target,
        class_id_to_name=cells_test['class_id_to_name'])



