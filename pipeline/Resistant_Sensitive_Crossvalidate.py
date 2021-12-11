from ProcessingPipeline import ProcessingPipeline as pipeline

import os
from pipeline.helpers import *
import numpy as np

import classification
import gc
from tensorflow.keras.backend import clear_session
from sklearn.metrics import ConfusionMatrixDisplay
from distutils.dir_util import copy_tree
import multiprocessing

from Resistant_Sensitive_Comparison import amend_class_labels

def crossvalidate_resistant_sensitive(output_path=None, resistant_experiments_path_list=None, sensitive_experiments_path_list=None, annotations_path=None, size_target=None,
                              pad_cells=False, resize_cells=False, class_count=None,
                              logdir=None, verbose=False, cond_IDs=None, image_channels=None, img_dims =None, mode=None,batch_size=None,learning_rate=None):





if __name__ == '__main__':
    output_path = os.path.join(get_parent_path(1), 'Data', 'Crossvalidate_06_12_21_registration_fixed_histeq_brightnessaug_misalign_gaussnoise_300perexp')
    cond_IDs = ['CIP+ETOH']
    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)

    annot_path = os.path.join(get_parent_path(1), 'Data', 'Segmentations_300PerExperiment_Improved_metric')

    experiment0 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_0_18_08_20')
    experiment1 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_1_25_03_21')
    experiment2 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_3_01_04_21')
    experiment3 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_4_03_04_21')
    experiment4 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_5_19_10_21')
    experiment5 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_6_25_10_21')

    experiments_path_list = [experiment0,experiment1,experiment2,experiment3,experiment4,experiment5]

    size_target = (64,64,3)

    logdir = os.path.join(get_parent_path(1),'Crossvalidate_06_12_21_registration_fixed_histeq_brightnessaug_misalign_gaussnoise_400perexp')

    crossvalidate_resistant_sensitive(output_path=output_path, experiments_path_list=experiments_path_list, annotations_path=annot_path, size_target=size_target,
                              pad_cells=True, resize_cells=False, class_count=3,
                              logdir=logdir, verbose=False, cond_IDs=cond_IDs, image_channels=image_channels, img_dims=img_dims, mode='DenseNet121',
                              batch_size=16, learning_rate=0.0005)