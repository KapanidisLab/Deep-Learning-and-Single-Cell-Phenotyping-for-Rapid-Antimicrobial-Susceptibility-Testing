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


def holdout_test(output_path=None, training_path_list=None, test_path = None, annotations_path=None, size_target=None,
                              pad_cells=False, resize_cells=False, class_count=None,
                              logdir=None, verbose=False, cond_IDs=None, image_channels=None, img_dims =None, mode=None,batch_size=None,learning_rate=None):

    #Make output folder
    makedir(output_path)

    # Fix pycharm console
    class PseudoTTY(object):
        def __init__(self, underlying):
            self.__underlying = underlying

        def __getattr__(self, name):
            return getattr(self.__underlying, name)

        def isatty(self):
            return True

    sys.stdout = PseudoTTY(sys.stdout)

    #Generate masks
    print('Generating masks.')


    p = pipeline(None,'NIM')

    for i in range(len(cond_IDs)):
        cond_ID = cond_IDs[i]
        corresponding_annotations = os.path.join(annotations_path,cond_ID)
        p.FileOp('masks_from_integer_encoding', mask_path=corresponding_annotations, output_path=corresponding_annotations)


    # Prepare train data
    output_segregated_train = os.path.join(output_path, 'Segregated_Train')
    output_collected_train = os.path.join(output_path, 'Collected_Train')

    local_pipeline_train = pipeline(training_path_list, 'NIM')
    local_pipeline_train.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
                              crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated_train)
    local_pipeline_train.path = output_segregated_train
    local_pipeline_train.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected_train,
                                 registration_target=0)

    data_sources = [os.path.join(output_collected_train, condition) for condition in cond_IDs]
    annotation_sources = [os.path.join(os.path.join(annotations_path, condition), 'annots') for condition in cond_IDs]

    dataset_output_train = os.path.join(output_path, 'Dataset_Train')

    local_pipeline_train.FileOp('TrainTestVal_split', data_sources=data_sources,
                                annotation_sources=annotation_sources, output_folder=dataset_output_train, test_size=0,
                                validation_size=0, seed=42)

    # Prepare test data
    output_segregated_test = os.path.join(output_path, 'Segregated_Test')
    output_collected_test = os.path.join(output_path, 'Collected_Test')

    local_pipeline_test = pipeline(test_path, 'NIM')
    local_pipeline_test.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
                             crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated_test)
    local_pipeline_test.path = output_segregated_test
    local_pipeline_test.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected_test,
                                registration_target=0)

    data_sources = [os.path.join(output_collected_test, condition) for condition in cond_IDs]

    dataset_output_test = os.path.join(output_path, 'Dataset_Test')

    local_pipeline_test.FileOp('TrainTestVal_split', data_sources=data_sources,
                               annotation_sources=annotation_sources, output_folder=dataset_output_test, test_size=1,
                               validation_size=0, seed=42)

    # Extract data
    manual_struct_train = classification.struct_from_file(
        dataset_folder=dataset_output_train,
        class_id=1)
    cells_train = classification.cells_from_struct(input=manual_struct_train, cond_IDs=cond_IDs,
                                                   image_dir=output_collected_train,
                                                   mode='masks')
    X_train, _, y_train, _ = classification.split_cell_sets(input=cells_train, test_size=0, random_state=42)

    manual_struct_test = classification.struct_from_file(
        dataset_folder=dataset_output_test,
        class_id=1)

    cells_test = classification.cells_from_struct(input=manual_struct_test, cond_IDs=cond_IDs,
                                                  image_dir=output_collected_test,
                                                  mode='masks')
    _, X_test, _, y_test = classification.split_cell_sets(input=cells_test, test_size=1, random_state=42)

    dt = 'MODE - {} BS - {} LR - {} Holdout test'.format(mode, batch_size, learning_rate)

    print()
    print('-------------------------------------')
    print('Training')
    print('-------------------------------------')
    print()

    kwargs = {'mode': mode, 'X_train': X_train, 'y_train': y_train, 'size_target': size_target, 'pad_cells': pad_cells,
              'resize_cells': resize_cells,
              'class_count': class_count, 'logdir': logdir, 'batch_size': batch_size, 'epochs': 100,
              'learning_rate': learning_rate, 'optimizer': 'NAdam',
              'verbose': verbose, 'dt_string': dt
              }

    p = multiprocessing.Process(target=classification.train, kwargs=kwargs)
    p.start()
    p.join()

    print()
    print('-------------------------------------')
    print('Evaluating on holdout')
    print('-------------------------------------')
    print()

    kwargs = {'modelpath': os.path.join(logdir, dt + '.h5'), 'X_test': X_test, 'y_test': y_test,
              'mean': np.asarray([0, 0, 0]),
              'size_target': size_target, 'pad_cells': pad_cells, 'resize_cells': resize_cells,
              'class_id_to_name': cells_train['class_id_to_name'],
              'normalise_CM': True, 'queue': None}

    p = multiprocessing.Process(target=classification.inspect, kwargs=kwargs)
    p.start()
    p.join()

if __name__ == '__main__':

    output_path = os.path.join(get_parent_path(1), 'Data', 'LabStrains_holdout')
    cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)

    annot_path = os.path.join(get_parent_path(1), 'Data', 'Segmentations_edgeremoved_300Perexperiment_newmetric')

    experiment0 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_0_18_08_20')
    experiment1 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_1_25_03_21')
    experiment2 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_3_01_04_21')
    experiment3 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_4_03_04_21')
    experiment4 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_5_19_10_21')
    experiment5 = os.path.join(get_parent_path(1), 'Data', 'Exp1', 'Repeat_6_25_10_21')

    holdout_experiment = os.path.join(get_parent_path(1), 'Data', 'Exp1_HoldOut_test', 'Repeat_7_01_12_21')

    experiments_path_list = [experiment0,experiment1,experiment2,experiment3,experiment4,experiment5]

    size_target = (64,64,3)

    logdir = output_path

    holdout_test(output_path = output_path, training_path_list = experiments_path_list, test_path = holdout_experiment, annotations_path = annot_path, size_target = size_target,
    pad_cells = True, resize_cells = False, class_count = 3,
    logdir = output_path, verbose = False, cond_IDs = cond_IDs, image_channels = image_channels, img_dims = img_dims, mode = 'DenseNet121', batch_size = 16, learning_rate = 0.0005)

