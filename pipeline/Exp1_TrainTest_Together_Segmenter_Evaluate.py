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

if __name__ == '__main__':

    data_folder = os.path.join(get_parent_path(1), 'Data', 'Exp1_HoldOut_Test', 'Repeat_7_01_12_21')

    output_segregated = os.path.join(get_parent_path(1), 'Data', 'Exp1_Segregated_Singlechannel')
    output_collected = os.path.join(get_parent_path(1), 'Data', 'Exp1_Collected_Singlechannel')
    output = os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp1_Singlechannel')

    cond_IDs = ['WT+ETOH','CIP+ETOH', 'RIF+ETOH']
    image_channels = ['NR', 'NR', 'NR']
    img_dims = (30, 684, 840)

    pipeline = ProcessingPipeline(data_folder, 'NIM')
    #pipeline.Sort(cond_IDs = cond_IDs, img_dims = img_dims, image_channels = image_channels, crop_mapping = {'DAPI':0, 'NR':0}, output_folder=output_segregated)
    #pipeline.Collect(cond_IDs = cond_IDs, image_channels = image_channels, output_folder = output_collected, registration_target=0)

    # ---GENERATE CELLS DATASET FROM SEGMENTATION MASKS AND BOTH CHANNELS, SPLIT AND SAVE. TRAIN AND TEST SEPARATELY

    input_path_WT = os.path.join(get_parent_path(1), 'Data', 'Segmentations_edgeremoved_300Perexperiment_newmetric','WT+ETOH')
    input_path_CIP = os.path.join(get_parent_path(1), 'Data', 'Segmentations_edgeremoved_300Perexperiment_newmetric', 'CIP+ETOH')
    input_path_RIF = os.path.join(get_parent_path(1), 'Data', 'Segmentations_edgeremoved_300Perexperiment_newmetric','RIF+ETOH')

    #pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_WT, output_path=input_path_WT)
    #pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_CIP, output_path= input_path_CIP)
    #pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_RIF, output_path=input_path_RIF)

    # --- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
    annots_WT = os.path.join(input_path_WT, 'annots')
    files_WT = os.path.join(output_collected, 'WT+ETOH')

    annots_CIP = os.path.join(input_path_CIP, 'annots')
    files_CIP = os.path.join(output_collected,'CIP+ETOH')

    annots_RIF = os.path.join(input_path_RIF, 'annots')
    files_RIF = os.path.join(output_collected, 'RIF+ETOH')



    pipeline.FileOp('TrainTestVal_split', data_sources=[files_WT, files_CIP, files_RIF], annotation_sources=[annots_WT, annots_CIP, annots_RIF],
                    output_folder=output, test_size=1, validation_size=0, seed=42)

    weights = r'C:\Users\zagajewski\Desktop\Deployment\mask_rcnn_EXP1.h5'
    test_dir = os.path.join(output,'Test')



    configuration = BacConfig()
    configuration.NAME = 'FirstStage1'
    evaluate_coco_metrics(dataset_folder=test_dir, config=configuration,
                          weights=weights)

#inspect_mrcnn_segmenter(test_folder = test_dir, configuration = configuration, weights = weights, ids=[0,20,30])