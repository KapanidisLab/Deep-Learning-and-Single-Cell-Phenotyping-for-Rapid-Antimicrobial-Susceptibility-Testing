from ProcessingPipeline import ProcessingPipeline

import os
from pipeline.helpers import *
import numpy as np

import classification
import gc
from tensorflow.keras.backend import clear_session
from sklearn.metrics import ConfusionMatrixDisplay
from distutils.dir_util import copy_tree
import multiprocessing
from classification import *


from Resistant_Sensitive_Comparison import amend_class_labels, combine_cells

def extract_resistant_sensitive_cells(resistant_paths=None,sensitive_paths=None, mode=None, cond_ID=None, annotations=None, output_path=None, image_channels=None,img_dims=None, resistant_strain_ID=None, sensitive_strain_ID=None):
    assert mode in ['train','test']
    assert cond_ID is not None
    assert annotations is not None

    # Process images
    output_resistant_segregated = os.path.join(output_path, resistant_strain_ID + '_' + mode + '_Segregated')
    output_resistant_collected = os.path.join(output_path, resistant_strain_ID + '_' + mode + '_Collected')

    output_sensitive_segregated = os.path.join(output_path, sensitive_strain_ID + '_' + mode +'_Segregated')
    output_sensitive_collected = os.path.join(output_path, sensitive_strain_ID + '_' + mode +'_Collected')

    pipeline_resistant = ProcessingPipeline(resistant_paths, 'NIM')
    #pipeline_resistant.Sort(cond_IDs=cond_ID, img_dims=img_dims, image_channels=image_channels,
     #                       crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_resistant_segregated)
    #pipeline_resistant.Collect(cond_IDs=cond_ID, image_channels=image_channels,
     #                          output_folder=output_resistant_collected, registration_target=0)

    pipeline_sensitive = ProcessingPipeline(sensitive_paths, 'NIM')
    #pipeline_sensitive.Sort(cond_IDs=cond_ID, img_dims=img_dims, image_channels=image_channels,
     #                       crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_sensitive_segregated)
    #pipeline_sensitive.Collect(cond_IDs=cond_ID, image_channels=image_channels,
     #                          output_folder=output_sensitive_collected, registration_target=0)


    # Prepare standard dataset objects
    dataset_output_resistant = os.path.join(output_path, 'Dataset' + '_' + resistant_strain_ID + '_' + mode)
    dataset_output_sensitive = os.path.join(output_path, 'Dataset' + '_' + sensitive_strain_ID + '_' + mode)

    data_source_resistant = os.path.join(output_resistant_collected, cond_ID[0])
    data_source_sensitive = os.path.join(output_sensitive_collected, cond_ID[0])

    #pipeline_resistant.FileOp('TrainTestVal_split', data_sources=[data_source_resistant],
     #                         annotation_sources=[annotations], output_folder=dataset_output_resistant,
      #                        test_size=1,
       #                       validation_size=0, seed=42)
    #pipeline_sensitive.FileOp('TrainTestVal_split', data_sources=[data_source_sensitive],
     #                         annotation_sources=[annotations], output_folder=dataset_output_sensitive,
      #                        test_size=1,
       #                       validation_size=0, seed=42)

    # Extract all cells
    resistant_struct = struct_from_file(dataset_folder=dataset_output_resistant, class_id=1)
    sensitive_struct = struct_from_file(dataset_folder=dataset_output_sensitive, class_id=1)

    resistant_cells = cells_from_struct(input=resistant_struct, cond_IDs=cond_ID, image_dir=output_resistant_collected,
                                        mode='masks')
    sensitive_cells = cells_from_struct(input=sensitive_struct, cond_IDs=cond_ID, image_dir=output_sensitive_collected,
                                        mode='masks')

    # Amend class labels of cells to match the comparison made
    resistant_cells = amend_class_labels(original_label=cond_ID[0], new_label=resistant_strain_ID, new_id=0,
                                         cells=resistant_cells)
    sensitive_cells = amend_class_labels(original_label=cond_ID[0], new_label=sensitive_strain_ID, new_id=1,
                                         cells=sensitive_cells)

    cells = combine_cells([resistant_cells, sensitive_cells])

    # Split sets
    if mode == 'train':
        X, _, y, _ = split_cell_sets(input=cells, test_size=0, random_state=42)
    elif mode == 'test':
        _, X, _, y = split_cell_sets(input=cells, test_size=1.0, random_state=42)

    return X,y, cells['class_id_to_name']

def crossvalidate_resistant_sensitive(output_path=None, experiments_path_list=None, annotations_path=None, size_target=None,
                              pad_cells=False, resize_cells=False,
                              logdir=None, verbose=False, cond_IDs=None, image_channels=None, img_dims =None, resistant_strain_ID=None, sensitive_strain_ID=None):

    #Make output folder
    makedir(output_path)

    output_path = os.path.join(output_path, resistant_strain_ID + '_' + sensitive_strain_ID)
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


    exp_count = len(experiments_path_list)

    Train_paths = []
    Test_paths = []

    print('Crossvalidating experiments: {} experiments supplied. Examining splits:'.format(exp_count))
    for i in range(exp_count):
        test = experiments_path_list[i]
        train = []

        for j in range(exp_count):
            if i == j:
                continue
            else:
                train.append(experiments_path_list[j])

        print('SPLIT {}'.format(i))
        print('TRAIN: ')
        for k in range(len(train)):
            print(train[k])
        print('TEST: ')
        print(test)

        print('')

        Train_paths.append(train)
        Test_paths.append(test)

    print('-------------------------------------------')
    assert len(Train_paths) == len(Test_paths)

    assert len(cond_IDs) == 1 and isinstance(cond_IDs, list), 'cond_ID must be a list with one entry.'

    #Generate masks
    print('Generating masks.')


    p = ProcessingPipeline(None,'NIM')

    for i in range(len(cond_IDs)):
        cond_ID = cond_IDs[i]
        corresponding_annotations = os.path.join(annotations_path,cond_ID)
        p.FileOp('masks_from_integer_encoding', mask_path=corresponding_annotations, output_path=corresponding_annotations)

    #Initialise CM for storage
    CM_total = np.zeros((len(cond_IDs),len(cond_IDs)))

    #Initialie mapping of class labels. As a sanity check, assert that this does not change
    class_mapping = None

    for i in range(len(Train_paths)):

        print()
        print('-------------------------------------')
        print('Preparing split {}'.format(i))
        print('-------------------------------------')
        print()

        #Prepare training cells
        traininig_paths = Train_paths[i]
        resistant_training_paths = [os.path.join(path, resistant_strain_ID) for path in traininig_paths ]
        sensitive_training_paths = [os.path.join(path, sensitive_strain_ID) for path in traininig_paths ]

        X_train,y_train, mapping_train = extract_resistant_sensitive_cells(resistant_paths=resistant_training_paths,
                                                                           sensitive_paths=sensitive_training_paths,
                                                                           mode='train', cond_ID=['CIP+ETOH'],
                                                                           annotations=os.path.join(corresponding_annotations,'annots'),
                                                                           output_path=output_path, img_dims=img_dims,image_channels=image_channels,
                                                                           resistant_strain_ID=resistant_strain_ID, sensitive_strain_ID=sensitive_strain_ID)

        #Prepare test cells
        testing_path = Test_paths[i]
        resistant_test_paths = os.path.join(testing_path, resistant_strain_ID)
        sensitive_test_paths = os.path.join(testing_path, sensitive_strain_ID)

        X_test, y_test, mapping_test = extract_resistant_sensitive_cells(resistant_paths=resistant_test_paths,
                                                             sensitive_paths=sensitive_test_paths, mode='test',
                                                             cond_ID=['CIP+ETOH'],
                                                             annotations=os.path.join(corresponding_annotations,'annots'),
                                                                         output_path=output_path, img_dims=img_dims,image_channels=image_channels,
                                                                         resistant_strain_ID=resistant_strain_ID, sensitive_strain_ID=sensitive_strain_ID)

        print('Length of test: ',len(X_test))
        print('length of train: ',len(X_train))

        assert mapping_test == mapping_train

        if class_mapping is None:
            class_mapping = mapping_train
        else:
            assert class_mapping == mapping_train == mapping_test

        # Train and Inspect
        dt_string = 'DenseNet121_CV_' + resistant_strain_ID + '_' + sensitive_strain_ID + '_' + str(i)
        class_count = 2
        logdir = output_path

        #Train split

        print()
        print('-------------------------------------')
        print('Training split {}'.format(i))
        print('-------------------------------------')
        print()


        kwargs = {'mode': 'DenseNet121', 'X_train': X_train, 'y_train': y_train, 'size_target':size_target, 'pad_cells':pad_cells, 'resize_cells':resize_cells,
                  'class_count':class_count, 'logdir':logdir, 'batch_size':16, 'epochs':100, 'learning_rate':0.0005, 'optimizer':'NAdam',
                  'verbose':verbose, 'dt_string':dt_string
                  }
        '''
        p = multiprocessing.Process(target=classification.train, kwargs=kwargs)
        p.start()
        p.join()
        '''

        #Evaluate split

        print()
        print('-------------------------------------')
        print('Evaluating split {}'.format(i))
        print('-------------------------------------')
        print()

        queue = multiprocessing.Queue()

        kwargs = {'modelpath': os.path.join(logdir, dt_string + '.h5'), 'X_test': X_test, 'y_test': y_test,
                  'mean': np.asarray([0, 0, 0]),
                  'size_target': size_target, 'pad_cells': pad_cells, 'resize_cells': resize_cells,
                  'class_id_to_name': class_mapping,
                  'normalise_CM': True, 'queue': queue}

        p = multiprocessing.Process(target=classification.inspect, kwargs=kwargs)
        p.start()
        CM_split = queue.get()
        p.join()

        CM_total = CM_total + CM_split

    #Map classnames to class labels
    labels = [0]*len(class_mapping) #initialise array
    for elm in class_mapping:
        labels[elm['class_id']] = elm['name']

    #Display final

    CM_normal = CM_total/np.sum(CM_total, axis=0)

    disp = ConfusionMatrixDisplay(confusion_matrix=CM_normal, display_labels = labels)
    disp.plot(cmap='Reds')
    plt.show()




if __name__ == '__main__':
    output_path = os.path.join(get_parent_path(1), 'Data', 'Clinical_Resistant_Sensitive_Crossvalidate_13_12_21')
    cond_IDs = ['CIP+ETOH']
    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)

    annot_path = os.path.join(get_parent_path(1), 'Data', 'Segmentations_Clinical_edgeremoved_300perexperiment_newmetric')

    size_target = (64,64,3)

    logdir = os.path.join(get_parent_path(1),'Data', 'Clinical_Resistant_Sensitive_Crossvalidate_13_12_21')

    experiment0 = os.path.join(get_parent_path(1), 'Data', 'Clinical_strains', 'Repeat_0_10_11_21+Repeat_1_18_11_21')
    experiment1= os.path.join(get_parent_path(1), 'Data', 'Clinical_strains', 'Repeat_2_03_12_21+Repeat_5_06_12_21')
    experiment2 = os.path.join(get_parent_path(1), 'Data', 'Clinical_strains', 'Repeat_3_04_12_21')

    experiments_path_list = [experiment0,experiment1,experiment2]

    ############################################ PAIRWISE 1 ##########################################################

    resistant_strain_ID = '13834'
    sensitive_strain_ID = '48480'

    kwargs = {'output_path':output_path, 'experiments_path_list':experiments_path_list, 'annotations_path':annot_path, 'size_target':size_target,
    'pad_cells':True, 'resize_cells':False,
    'logdir':logdir, 'verbose':False, 'cond_IDs':['CIP+ETOH'], 'image_channels':image_channels, 'img_dims':img_dims, 'resistant_strain_ID':resistant_strain_ID, 'sensitive_strain_ID':sensitive_strain_ID}

    p = multiprocessing.Process(target=crossvalidate_resistant_sensitive, kwargs=kwargs)
    p.start()
    p.join()

    ############################################ PAIRWISE 2 ##########################################################

    resistant_strain_ID = '13834'
    sensitive_strain_ID = '64017'

    kwargs = {'output_path':output_path, 'experiments_path_list':experiments_path_list, 'annotations_path':annot_path, 'size_target':size_target,
    'pad_cells':True, 'resize_cells':False,
    'logdir':logdir, 'verbose':False, 'cond_IDs':['CIP+ETOH'], 'image_channels':image_channels, 'img_dims':img_dims, 'resistant_strain_ID':resistant_strain_ID, 'sensitive_strain_ID':sensitive_strain_ID}

    p = multiprocessing.Process(target=crossvalidate_resistant_sensitive, kwargs=kwargs)
    p.start()
    p.join()

    ############################################ PAIRWISE 3 ##########################################################

    resistant_strain_ID = '17667'
    sensitive_strain_ID = '64017'

    kwargs = {'output_path':output_path, 'experiments_path_list':experiments_path_list, 'annotations_path':annot_path, 'size_target':size_target,
    'pad_cells':True, 'resize_cells':False,
    'logdir':logdir, 'verbose':False, 'cond_IDs':['CIP+ETOH'], 'image_channels':image_channels, 'img_dims':img_dims, 'resistant_strain_ID':resistant_strain_ID, 'sensitive_strain_ID':sensitive_strain_ID}

    p = multiprocessing.Process(target=crossvalidate_resistant_sensitive, kwargs=kwargs)
    p.start()
    p.join()

    ############################################ PAIRWISE 4 ##########################################################

    resistant_strain_ID = '17667'
    sensitive_strain_ID = '48480'

    kwargs = {'output_path':output_path, 'experiments_path_list':experiments_path_list, 'annotations_path':annot_path, 'size_target':size_target,
    'pad_cells':True, 'resize_cells':False,
    'logdir':logdir, 'verbose':False, 'cond_IDs':['CIP+ETOH'], 'image_channels':image_channels, 'img_dims':img_dims, 'resistant_strain_ID':resistant_strain_ID, 'sensitive_strain_ID':sensitive_strain_ID}

    p = multiprocessing.Process(target=crossvalidate_resistant_sensitive, kwargs=kwargs)
    p.start()
    p.join()