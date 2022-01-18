from ProcessingPipeline import ProcessingPipeline
import os
from helpers import *
import tensorflow as tf

from classification import *

from similarity import train_triplet_similarity

if __name__ == '__main__':

    data_folder = os.path.join(get_parent_path(1), 'Data', 'Exp1')

    output=os.path.join(get_parent_path(1),'Data','06_01_22_Similarity')
    makedir(output)

    output_segregated = os.path.join(output, 'Segregated')
    output_collected = os.path.join(output, 'Collected')
    output_dataset = os.path.join(output,'Dataset')

    cond_IDs = ['WT+ETOH', 'CIP+ETOH']
    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)

    resize_target = (64,64,3)
    #backbone_weights = r'C:\Users\zagajewski\Desktop\Deployment\WT0_CIP1_holdout.h5'
    backbone_weights = None
    modelname = 'Test1'

    # Fix pycharm console
    class PseudoTTY(object):
        def __init__(self, underlying):
            self.__underlying = underlying

        def __getattr__(self, name):
            return getattr(self.__underlying, name)

        def isatty(self):
            return True


    sys.stdout = PseudoTTY(sys.stdout)

    pipeline = ProcessingPipeline(data_folder, 'NIM')
    #pipeline.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
     #             crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated)
    #pipeline.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected,
      #               registration_target=0)

    # ---GENERATE CELLS DATASET FROM SEGMENTATION MASKS AND BOTH CHANNELS, SPLIT AND SAVE. TRAIN AND TEST SEPARATELY

    input_path_WT = os.path.join(get_parent_path(1), 'Data', 'Segmentations_Edge_Removed', 'WT+ETOH')
    input_path_CIP = os.path.join(get_parent_path(1), 'Data', 'Segmentations_Edge_Removed', 'CIP+ETOH')

    #pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_WT, output_path = input_path_WT)
    #pipeline.FileOp('masks_from_integer_encoding', mask_path=input_path_CIP, output_path= input_path_CIP)

    # --- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
    annots_WT = os.path.join(input_path_WT, 'annots')
    files_WT = os.path.join(output_collected, 'WT+ETOH')

    annots_CIP = os.path.join(input_path_CIP, 'annots')
    files_CIP = os.path.join(output_collected,'CIP+ETOH')


    pipeline.FileOp('TrainTestVal_split', data_sources=[files_WT, files_CIP],
                    annotation_sources=[annots_WT, annots_CIP], output_folder=output_dataset, test_size=0.2,
                    validation_size=0.2, seed=42)

    #Find segmentations
    manual_struct = struct_from_file(
        dataset_folder=output_dataset,
        class_id=1)

    #Extract cells and match with labels
    cells = cells_from_struct(input=manual_struct, cond_IDs=cond_IDs, image_dir=output_collected, mode='masks')
    X_train, _, y_train, _ = split_cell_sets(input=cells, test_size=0, random_state=42)

    with tf.InteractiveSession().as_default() as session:
        train_triplet_similarity(backbone_weights=backbone_weights, cells=X_train, labels=y_train, target_shape=resize_target, freeze_backbone=False,encoder_version=1, optimizer='NAdam', initial_lr=0.0001, batch_size=256, dt_string=modelname, logdir=output)