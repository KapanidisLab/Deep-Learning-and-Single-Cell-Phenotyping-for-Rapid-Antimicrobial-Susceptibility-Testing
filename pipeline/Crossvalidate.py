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
import seaborn as sns
from Resistant_Sensitive_Comparison import amend_class_labels
from Lab_strains_hold_out_test import select_n_by_experiment


def crossvalidate_experiments(output_path=None, experiments_path_list=None, annotations_path=None, size_target=None,
                              pad_cells=False, resize_cells=False, class_count=None,
                              logdir=None, verbose=False, cond_IDs=None, image_channels=None, img_dims =None, mode=None,batch_size=None,learning_rate=None,
                              colour_mapping=None,cells_per_experiment=None,epochs=None, optimizer=None,transfer_weights=None):

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

    #Generate masks
    print('Generating masks.')


    p = pipeline(None,'NIM')

    for i in range(len(cond_IDs)):
        cond_ID = cond_IDs[i]
        corresponding_annotations = os.path.join(annotations_path,cond_ID)
#        p.FileOp('masks_from_integer_encoding', mask_path=corresponding_annotations, output_path=corresponding_annotations)

    #Initialise CM for storage
    CM_total = np.zeros((len(cond_IDs),len(cond_IDs)))

    for i in range(len(Train_paths)):

        print()
        print('-------------------------------------')
        print('Preparing split {}'.format(i))
        print('-------------------------------------')
        print()


        #Prepare train data
        split_path = os.path.join(output_path,str(i))
        makedir(split_path)

        output_segregated_train = os.path.join(split_path,'Segregated_Train')
        output_collected_train = os.path.join(split_path,'Collected_Train')

        train_folder = Train_paths[i]



        local_pipeline_train = pipeline(train_folder, 'NIM')
        local_pipeline_train.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
                            crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated_train)
        local_pipeline_train.path = output_segregated_train
        local_pipeline_train.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected_train,
                               registration_target=0)

        data_sources = [os.path.join(output_collected_train,condition) for condition in cond_IDs]
        annotation_sources = [os.path.join(os.path.join(annotations_path,condition),'annots') for condition in cond_IDs]

        dataset_output_train = os.path.join(split_path,'Dataset_Train')

        local_pipeline_train.FileOp('TrainTestVal_split', data_sources=data_sources,
                             annotation_sources=annotation_sources, output_folder=dataset_output_train, test_size=0,
                             validation_size=0, seed=42)


        #Prepare test data
        test_folder = Test_paths[i]

        output_segregated_test = os.path.join(split_path,'Segregated_Test')
        output_collected_test = os.path.join(split_path,'Collected_Test')

        local_pipeline_test = pipeline(test_folder, 'NIM')
        local_pipeline_test.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
                            crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated_test)
        local_pipeline_test.path = output_segregated_test
        local_pipeline_test.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected_test,
                              registration_target=0)

        data_sources = [os.path.join(output_collected_test, condition) for condition in cond_IDs]

        dataset_output_test = os.path.join(split_path, 'Dataset_Test')

        local_pipeline_test.FileOp('TrainTestVal_split', data_sources=data_sources,
                             annotation_sources=annotation_sources, output_folder=dataset_output_test, test_size=1,
                             validation_size=0, seed=42)


        #Extract data

        manual_struct_train = classification.struct_from_file(
            dataset_folder=dataset_output_train,
            class_id=1)

        cells_train = select_n_by_experiment(struct=manual_struct_train, cond_IDs=cond_IDs,
                                             image_dir=output_collected_train, n=cells_per_experiment)

        cells_train = amend_class_labels(original_label='WT+ETOH',new_label='Untreated',new_id=0,cells=cells_train)

        if 'CIP+ETOH' in cond_IDs:
            cells_train = amend_class_labels(original_label='CIP+ETOH', new_label='CIP', new_id=1, cells=cells_train)
        elif 'RIF+ETOH' in cond_IDs:
            cells_train = amend_class_labels(original_label='RIF+ETOH', new_label='RIF', new_id=1, cells=cells_train)
        elif 'GENT+ETOH' in cond_IDs:
            cells_train = amend_class_labels(original_label='GENT+ETOH', new_label='GENT', new_id=1, cells=cells_train)
        elif 'CEFT+ETOH' in cond_IDs:
            cells_train = amend_class_labels(original_label='CEFT+ETOH', new_label='CEFT', new_id=1, cells=cells_train)
        elif 'COAMOX+ETOH' in cond_IDs:
            cells_train = amend_class_labels(original_label='COAMOX+ETOH', new_label='COAMOX', new_id=1, cells=cells_train)

        else:
            raise ValueError()


        X_train, _, y_train, _ = classification.split_cell_sets(input=cells_train, test_size=0, random_state=42)

        manual_struct_test = classification.struct_from_file(
            dataset_folder=dataset_output_test,
            class_id=1)

        cells_test = select_n_by_experiment(struct=manual_struct_test, cond_IDs=cond_IDs,
                                             image_dir=output_collected_test, n=cells_per_experiment)

        cells_test = amend_class_labels(original_label='WT+ETOH',new_label='Untreated',new_id=0,cells=cells_test)

        if 'CIP+ETOH' in cond_IDs:
            cells_test = amend_class_labels(original_label='CIP+ETOH', new_label='CIP', new_id=1, cells=cells_test)
        elif 'RIF+ETOH' in cond_IDs:
            cells_test = amend_class_labels(original_label='RIF+ETOH', new_label='RIF', new_id=1, cells=cells_test)
        elif 'GENT+ETOH' in cond_IDs:
            cells_test = amend_class_labels(original_label='GENT+ETOH', new_label='GENT', new_id=1, cells=cells_test)
        elif 'CEFT+ETOH' in cond_IDs:
            cells_test = amend_class_labels(original_label='CEFT+ETOH', new_label='CEFT', new_id=1, cells=cells_test)
        elif 'COAMOX+ETOH' in cond_IDs:
            cells_test = amend_class_labels(original_label='COAMOX+ETOH', new_label='COAMOX', new_id=1, cells=cells_test)

        else:
            raise ValueError()

        _, X_test, _, y_test = classification.split_cell_sets(input=cells_test, test_size=1, random_state=42)

        dt = 'MODE - {} BS - {} LR - {} SPLIT - {}'.format(mode,batch_size,learning_rate,i)
        logdir = split_path

        print()
        print('-------------------------------------')
        print('Training split {}'.format(i))
        print('-------------------------------------')
        print()


        kwargs = {'mode': mode, 'X_train': X_train, 'y_train': y_train, 'size_target':size_target, 'pad_cells':pad_cells, 'resize_cells':resize_cells,
                  'class_count':class_count, 'logdir':logdir, 'batch_size':batch_size, 'epochs':epochs, 'learning_rate':learning_rate, 'optimizer':optimizer,
                  'verbose':verbose, 'dt_string':dt, 'init_source':transfer_weights
                  }

        p = multiprocessing.Process(target=classification.train, kwargs=kwargs)
        p.start()
        p.join()


        print()
        print('-------------------------------------')
        print('Evaluating split {}'.format(i))
        print('-------------------------------------')
        print()

        queue = multiprocessing.Queue()
        kwargs = {'modelpath':os.path.join(logdir, dt+'.h5'), 'X_test':X_test, 'y_test':y_test,'mean':np.asarray([0, 0, 0]),
                  'size_target':size_target, 'pad_cells':pad_cells, 'resize_cells':resize_cells, 'class_id_to_name':cells_train['class_id_to_name'],
                  'normalise_CM':False, 'queue':queue, 'colour_mapping': colour_mapping}

        p = multiprocessing.Process(target=classification.inspect, kwargs=kwargs)
        p.start()
        CM_split = queue.get()
        p.join()

        CM_total = CM_total+CM_split

    #Map classnames to class labels
    labels = [0]*len(cells_train['class_id_to_name']) #initialise array
    colour_mask = [0]*len(cells_train['class_id_to_name'])
    for elm in cells_train['class_id_to_name']:
        labels[elm['class_id']] = elm['name']
        colour_mask[elm['class_id']] = colour_mapping[elm['name']]

    #Display final

    CM_normal = CM_total/np.sum(CM_total, axis=1) #Final percentage
    CM_percentage = 100 * CM_normal #Raw category counts


    CM_total = np.asarray(np.rint(CM_total),dtype='int')
    processed_counts = CM_total.flatten().tolist()
    processed_counts = ['({})'.format(elm) for elm in processed_counts]
    processed_counts = np.asarray(processed_counts).reshape((2, 2))

    processed_percentage = np.asarray(np.rint(CM_percentage.flatten()), dtype='int').tolist()
    processed_percentage = ['{}%'.format(elm) for elm in processed_percentage]
    processed_percentage = np.asarray(processed_percentage).reshape((2, 2))

    formatted_text = (np.asarray(["{}\n\n{}".format(
        data, text) for text, data in zip(processed_counts.flatten(), processed_percentage.flatten())])).reshape(2, 2)

    sns.set(font_scale=2.0)
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    plt.tight_layout()
    for i in range(len(colour_mask)):

        mask = np.ones(CM_percentage.shape)
        mask[:,i] = 0
        sns.heatmap(CM_percentage,linewidths=2, linecolor="black",  ax=ax,annot=formatted_text, cbar=False, vmin=0,vmax=100, fmt='',cmap=colour_mask[i], mask=mask)


    # labels, title and ticks
    ax.set_xlabel('Predicted labels', fontsize=20)
    ax.set_ylabel('True labels', fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=20)
    ax.yaxis.set_ticklabels(labels, fontsize=20)

    ax.axhline(y=0, color='k', linewidth=3)
    ax.axhline(y=CM_percentage.shape[1], color='k', linewidth=3)
    ax.axvline(x=0, color='k', linewidth=3)
    ax.axvline(x=CM_percentage.shape[0], color='k', linewidth=3)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    experiment0 = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Exp1\Repeat_0_18_08_20'
    experiment1 = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Exp1\Repeat_1_25_03_21'
    experiment2 = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Exp1\Repeat_3_01_04_21'
    experiment3 = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Exp1\Repeat_4_03_04_21'
    experiment4 = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Exp1\Repeat_5_19_10_21'
    experiment5 = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Exp1\Repeat_6_25_10_21'

    annot_path = os.path.join(get_parent_path(1), 'Data', 'Conor_DFP_Segmentations_all')

    image_channels = ['NR', 'DAPI']
    img_dims = ((30,1), 684, (840,856))
    experiments_path_list = [experiment0,experiment1,experiment2,experiment3,experiment4,experiment5]
    size_target = (64,64,3)

    transfer_weights = r'C:\Users\zagajewski\Desktop\AMR_ms_data_models\WT0CIP1_Holdout_Test\DenseNet121 BS - 16 LR - 0.0005 Holdout test.h5'

    #--------------------------------------------------WT_RIF-----------------------------------------------------------

    output_path = os.path.join(get_parent_path(1), 'Data','Crossvalidate_WTCIP_No_Background_Masking')
    cond_IDs = ['WT+ETOH', 'CIP+ETOH']

    logdir = output_path

    crossvalidate_experiments(output_path=output_path, experiments_path_list=experiments_path_list, annotations_path=annot_path, size_target=size_target,
                              pad_cells=True, resize_cells=False, class_count=2,
                              logdir=logdir, verbose=True, cond_IDs=cond_IDs, image_channels=image_channels, img_dims=img_dims, mode='DenseNet121',
                              batch_size=16, learning_rate=0.0005, colour_mapping={'Untreated':sns.light_palette((0, 75, 60), input="husl"), 'CIP':sns.light_palette((260, 75, 60), input="husl")}, cells_per_experiment=500, epochs=100, optimizer='NAdam')


    #--------------------------------------------------WT_GENT-----------------------------------------------------------

    #output_path = os.path.join(get_parent_path(1), 'Data','Crossvalidate_WT_GENT')
    #cond_IDs = ['WT+ETOH', 'GENT+ETOH']

    #logdir = os.path.join(get_parent_path(1),'Data','Crossvalidate_WT_GENT')

    #crossvalidate_experiments(output_path=output_path, experiments_path_list=experiments_path_list, annotations_path=annot_path, size_target=size_target,
     #                         pad_cells=True, resize_cells=False, class_count=2,
      #                        logdir=logdir, verbose=False, cond_IDs=cond_IDs, image_channels=image_channels, img_dims=img_dims, mode='DenseNet121',
       #                       batch_size=8, learning_rate=0.001, colour_mapping={'Untreated':sns.light_palette((0, 75, 60), input="husl"), 'GENT':sns.light_palette((260, 75, 60), input="husl")},cells_per_experiment=500, epochs=100, optimizer='NAdam')