from ProcessingPipeline import ProcessingPipeline
import os
from helpers import *
from Resistant_Sensitive_Comparison import amend_class_labels
from Lab_strains_hold_out_test import select_n_cells

from classification import *

if __name__ == '__main__':
    # ---PREPARE DATASET WITH BOTH CHANNELS

    #data_folder = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Conor_DFP_Data_training'
    data_folder = None

    output_folder = os.path.join(get_parent_path(1), 'Data', 'WT0GENT1_Ensemble')

    makedir(output_folder)

    output_segregated = os.path.join(output_folder, 'Segregated_Multichannel')
    output_collected = os.path.join(output_folder, 'Collected_Multichannel')
    output_dataset = os.path.join(output_folder, 'Dataset_Multichannel')

    dt_string = 'DenseNet121_WTGENT_ENSEMBLE'

    cond_IDs = ['WT+ETOH', 'GENT+ETOH']
    image_channels = ['NR', 'DAPI']
    img_dims = ((30,1), 684, (856,840))

    pipeline = ProcessingPipeline(data_folder, 'NIM')
    #pipeline.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
     #             crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated)
    pipeline.path=output_segregated
    pipeline.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected,
                     registration_target=0)

    # ---GENERATE CELLS DATASET FROM SEGMENTATION MASKS AND BOTH CHANNELS, SPLIT AND SAVE. TRAIN AND TEST SEPARATELY
    input_path_WT = os.path.join(get_parent_path(1), 'Data', 'New_antibiotics_segmentations_all', 'WT+ETOH')
    input_path_GENT = os.path.join(get_parent_path(1), 'Data', 'New_antibiotics_segmentations_all', 'GENT+ETOH')

    #pipeline.FileOp('masks_from_integer_encoding_CF', mask_path=input_path_WT, output_path = input_path_WT)
    #pipeline.FileOp('masks_from_integer_encoding_CF', mask_path=input_path_CIP, output_path= input_path_CIP)

    # --- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
    annots_WT = os.path.join(input_path_WT, 'annots')
    files_WT = os.path.join(output_collected, 'WT+ETOH')

    annots_GENT = os.path.join(input_path_GENT, 'annots')
    files_GENT = os.path.join(output_collected, 'GENT+ETOH')

    pipeline.FileOp('TrainTestVal_split', data_sources=[files_WT, files_GENT],
                    annotation_sources=[annots_WT, annots_GENT], output_folder=output_dataset, test_size=0.2,
                    validation_size=0.2, seed=42)

    manual_struct = struct_from_file(dataset_folder=output_dataset,
                                     class_id=1)
    #Remove edge cells
    manual_struct,removed = remove_edge_cells(manual_struct)  # Remove edge cells
    print('Removed {} edge cells'.format(removed))

    #Generate cells
    cells = cells_from_struct(input=manual_struct, cond_IDs=cond_IDs, image_dir=output_collected, mode='masks')

    #Select 10k random non-edge cells from each class
    #cells = select_n_cells(cells, 10000)

    # Amend label names for nicer display
    cells = amend_class_labels(original_label='GENT+ETOH', new_label='GENT', new_id=1, cells=cells)
    cells = amend_class_labels(original_label='WT+ETOH', new_label='Untreated', new_id=0, cells=cells)

    X_train, X_test, y_train, y_test = split_cell_sets(input=cells, test_size=0.2, random_state=42)


    # Fix pycharm console
    class PseudoTTY(object):
        def __init__(self, underlying):
            self.__underlying = underlying

        def __getattr__(self, name):
            return getattr(self.__underlying, name)

        def isatty(self):
            return True


    sys.stdout = PseudoTTY(sys.stdout)

    # ---TRAIN---
    resize_target = (64, 64, 3)
    class_count = 2
    pad_cells = True
    resize_cells = False

    parameter_grid = {'batch_size': [8,16,32,64], 'learning_rate': [0.0005,0.001,0.002], 'optimizer' :['NAdam', 'SGD'], 'epochs':[100]}
    #optimize(mode = 'DenseNet121', X_train = X_train, y_train = y_train, parameter_grid = parameter_grid, size_target = resize_target, pad_cells = pad_cells, class_count = class_count, logdir = output_folder, resize_cells=resize_cells)

    train(mode='DenseNet121', X_train=X_train, y_train=y_train, size_target=resize_target, class_count=class_count,
          pad_cells=True,
          logdir=output_folder, batch_size=8, epochs=100, learning_rate=0.001, optimizer='NAdam', verbose=True,
          dt_string=dt_string)
    inspect(modelpath=os.path.join(output_folder, dt_string + '.h5'), X_test=X_test, y_test=y_test,
            mean=np.asarray([0, 0, 0]), size_target=resize_target, pad_cells=True,
            class_id_to_name=cells['class_id_to_name'], colour_mapping={'Untreated':sns.light_palette((0, 75, 60), input="husl"), 'GENT':sns.light_palette((260, 75, 60), input="husl")})

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------