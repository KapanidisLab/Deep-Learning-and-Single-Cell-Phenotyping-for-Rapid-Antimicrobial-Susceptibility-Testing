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

data_folder = os.path.join(get_parent_path(1),'Data','Exp1')
output_segregated = os.path.join(get_parent_path(1),'Data','Exp1_Segregated')
output_collected = os.path.join(get_parent_path(1),'Data','Exp1_Collected')

cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']

image_channels = ['NR','NR','NR']
img_dims = (30,684,840)

pipeline = ProcessingPipeline(data_folder, 'NIM')

pipeline.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0,'NR':0}, output_folder=output_segregated)
pipeline.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected, registration_target=None)

input_path_WT = os.path.join(get_parent_path(1), 'Data', 'Segmentations_All', 'WT+ETOH')
input_path_CIP = os.path.join(get_parent_path(1), 'Data', 'Segmentations_All', 'CIP+ETOH')
input_path_RIF = os.path.join(get_parent_path(1), 'Data', 'Segmentations_All', 'RIF+ETOH')

pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_WT, output_path = input_path_WT)
pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_CIP, output_path= input_path_CIP)
pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_RIF, output_path= input_path_RIF)


#--- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
annots_WT = os.path.join(input_path_WT, 'annots')
files_WT = os.path.join(output_collected,'WT+ETOH')

annots_CIP = os.path.join(input_path_CIP, 'annots')
files_CIP = os.path.join(output_collected,'CIP+ETOH')

annots_RIF = os.path.join(input_path_RIF, 'annots')
files_RIF = os.path.join(output_collected,'RIF+ETOH')

output = os.path.join(get_parent_path(1),'Data', 'Dataset_Exp1')


pipeline.FileOp('TrainTestVal_split', data_sources = [files_WT,files_CIP,files_RIF], annotation_sources = [annots_WT,annots_CIP,annots_RIF], output_folder=output,test_size = 0.2, validation_size=0.2, seed=42 )

#---TRAIN 1ST STAGE MODEL---

weights_start = os.path.join(get_parent_path(1), 'Data','MaskRCNN_pretrained_coco.h5')
train_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp1', 'Train')
val_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp1', 'Validation')
test_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp1', 'Test')
output_dir = get_parent_path(1)

configuration = BacConfig()
configuration.NAME = 'FirstStage1'

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

# --- TRAIN 1st STAGE SEGMENTER


#inspect_dataset(dataset_folder = train_dir)
#inspect_augmentation(dataset_folder = train_dir, configuration = configuration, augmentation = augmentation)

#Fix pycharm console
class PseudoTTY(object):
    def __init__(self, underlying):
        self.__underlying = underlying
    def __getattr__(self, name):
        return getattr(self.__underlying, name)
    def isatty(self):
        return True

sys.stdout = PseudoTTY(sys.stdout)

train_mrcnn_segmenter(train_folder = train_dir, validation_folder = val_dir, configuration = configuration, augmentation = augmentation, weights = weights_start, output_folder = output_dir)




