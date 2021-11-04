
import os
from ProcessingPipeline import ProcessingPipeline
import os
from helpers import *
from helpers import *
from implementations import *
from mask_generators import *
from segmentation import *
from classification import *

from classification import *

''''''

data_folder_train = os.path.join(get_parent_path(1),'Data','Train_0+3')
output_segregated_train = os.path.join(get_parent_path(1),'Data','Train_0+3_singlechannel_segregated')
output_collected_train = os.path.join(get_parent_path(1),'Data','Train_0+3_singlechannel_collected')

data_folder_test = os.path.join(get_parent_path(1),'Data','Test_4')
output_segregated_test = os.path.join(get_parent_path(1),'Data','Test_4_singlechannel_segregated')
output_collected_test = os.path.join(get_parent_path(1),'Data','Test_4_singlechannel_collected')


cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH', 'KAN+ETOH', 'CARB+ETOH']
image_channels = ['NR','DAPI']
img_dims = (30,684,840)

pipeline = ProcessingPipeline(data_folder_train, 'NIM')
pipeline.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0,'NR':0}, output_folder=output_segregated_train)
pipeline.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected_train, registration_target=None)

pipeline2 = ProcessingPipeline(data_folder_test,'NIM')
pipeline2.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0,'NR':0}, output_folder=output_segregated_test)
pipeline2.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected_test, registration_target=None)

'''

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

evaluate_coco_metrics(dataset_folder=test_dir, config=configuration,
                      weights=os.path.join(get_parent_path(1), "predconfig_1+320211013T2353",
                                           "mask_rcnn_predconfig_1+3.h5"))

# --- TRAIN 1st STAGE SEGMENTER

#train_mrcnn_segmenter(train_folder = train_dir, validation_folder = val_dir, configuration = configuration, augmentation = augmentation, weights = weights_start, output_folder = output_dir)

# --- INSPECT 1st STAGE STEPWISE AND OPTIMISE

#inspect_segmenter_stepwise(train_folder = train_dir, test_folder = test_dir, configuration = configuration, weights = weights)
#optimise_mrcnn_segmenter(mode = 'training', arg_names = ['LEARNING_RATE', 'IMAGES_PER_GPU'], arg_values = [[0.007,0.01],[4,6,8]], train_folder = train_dir, validation_folder = val_dir, configuration = configuration, augmentation = augmentation, weights = weights_start, output_folder = output_dir )
#optimise_mrcnn_segmenter(mode = 'inference', arg_names = ['DETECTION_NMS_THRESHOLD' ], arg_values = [[0.2,0.1]], test_folder=test_dir, configuration=configuration, weights=weights, ids=ids)
#inspect_mrcnn_segmenter(test_folder = test_dir, configuration = configuration, weights = weights, ids=ids )
