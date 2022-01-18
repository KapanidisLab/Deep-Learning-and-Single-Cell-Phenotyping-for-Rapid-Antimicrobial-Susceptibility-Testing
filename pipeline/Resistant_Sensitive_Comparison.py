import copy
import os
from helpers import *
from ProcessingPipeline import ProcessingPipeline
from classification import *

def amend_class_labels(original_label=None, new_label=None, new_id=None, cells=None):
    assert isinstance(cells,dict)
    assert 'class_id_to_name' in cells
    assert isinstance(new_id,int)

    label_found = False

    #update mapping
    for mapping in cells['class_id_to_name']:
        if not mapping['name'] == original_label: continue
        else:
            label_found = True

            mapping['class_id'] = new_id
            mapping['name'] = new_label

    assert label_found, 'Original label not found in cell mapping, double check and try again.'

    #update data name
    assert original_label in cells, 'Original label not found in cells, double check and try again.'

    cells[new_label] = cells.pop(original_label)

    return cells

def combine_cells(cells_list):
    output = {}
    new_mapping = []

    #concatenate mappings
    for cells in cells_list:
        for mapping_element in cells['class_id_to_name']:
            new_mapping.append(mapping_element)

        for key,value in cells.items():
            if key == 'class_id_to_name': continue
            else:
                output[key] = copy.deepcopy(value)

    output['class_id_to_name'] = new_mapping

    return output





def resistant_sensitive_comparison(output_path = None, cond_ID = None, image_channels = None, img_dims = None, resistant_path=None, sensitive_path=None, resistant_strain_ID = None, sensitive_strain_ID = None, annotations_path = None):
    # Make output folder
    makedir(output_path)

    output_path = os.path.join(output_path, resistant_strain_ID+'_'+sensitive_strain_ID)
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

    assert len(cond_ID) == 1 and isinstance(cond_ID,list), 'cond_ID must be a list with one entry.'

    #Process images
    output_resistant_segregated = os.path.join(output_path,resistant_strain_ID+'_Segregated')
    output_resistant_collected = os.path.join(output_path,resistant_strain_ID+'_Collected')

    output_sensitive_segregated = os.path.join(output_path,sensitive_strain_ID+'_Segregated')
    output_sensitive_collected = os.path.join(output_path,sensitive_strain_ID+'_Collected')

    pipeline_resistant = ProcessingPipeline(resistant_path, 'NIM')
    pipeline_resistant.Sort(cond_IDs=cond_ID, img_dims=img_dims, image_channels=image_channels,
                  crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_resistant_segregated)
    pipeline_resistant.Collect(cond_IDs = cond_ID, image_channels = image_channels, output_folder = output_resistant_collected, registration_target=0)


    pipeline_sensitive = ProcessingPipeline(sensitive_path, 'NIM')
    pipeline_sensitive.Sort(cond_IDs=cond_ID, img_dims=img_dims, image_channels=image_channels,
                  crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_sensitive_segregated)
    pipeline_sensitive.Collect(cond_IDs = cond_ID, image_channels = image_channels, output_folder = output_sensitive_collected, registration_target=0)

    #Extract masks
    corresponding_annotations = os.path.join(annotations_path, cond_ID[0])
    pipeline_sensitive.FileOp('masks_from_integer_encoding', mask_path=corresponding_annotations, output_path=corresponding_annotations)
    corresponding_annotations = os.path.join(corresponding_annotations,'annots')

    #Prepare standard dataset objects
    dataset_output_resistant = os.path.join(output_path,'Dataset_'+resistant_strain_ID)
    dataset_output_sensitive = os.path.join(output_path,'Dataset_'+sensitive_strain_ID)

    data_source_resistant = os.path.join(output_resistant_collected,cond_ID[0])
    data_source_sensitive = os.path.join(output_sensitive_collected,cond_ID[0])

    pipeline_resistant.FileOp('TrainTestVal_split', data_sources=[data_source_resistant],
                        annotation_sources=[corresponding_annotations], output_folder=dataset_output_resistant, test_size=1,
                       validation_size=0, seed=42)
    pipeline_sensitive.FileOp('TrainTestVal_split', data_sources=[data_source_sensitive],
                        annotation_sources=[corresponding_annotations], output_folder=dataset_output_sensitive, test_size=1,
                       validation_size=0, seed=42)


    #Extract all cells
    resistant_struct = struct_from_file(dataset_folder=dataset_output_resistant,class_id=1)
    sensitive_struct = struct_from_file(dataset_folder=dataset_output_sensitive,class_id=1)

    resistant_cells = cells_from_struct(input=resistant_struct, cond_IDs=cond_ID, image_dir=output_resistant_collected, mode='masks')
    sensitive_cells = cells_from_struct(input=sensitive_struct, cond_IDs=cond_ID, image_dir=output_sensitive_collected, mode='masks')

    #Amend class labels of cells to match the comparison made
    resistant_cells = amend_class_labels(original_label=cond_ID[0], new_label=resistant_strain_ID, new_id=0, cells=resistant_cells)
    sensitive_cells = amend_class_labels(original_label=cond_ID[0], new_label=sensitive_strain_ID, new_id=1, cells=sensitive_cells)

    cells = combine_cells([resistant_cells,sensitive_cells])

    #Split sets
    X_train, X_test, y_train, y_test = split_cell_sets(input=cells, test_size=0.2, random_state=42)

    #Train and Inspect
    dt_string = 'DenseNet121_'+resistant_strain_ID+'_'+sensitive_strain_ID
    size_target = (64, 64, 3)
    class_count = 2
    logdir = output_path


    train(mode='DenseNet121', X_train=X_train, y_train=y_train, size_target=size_target, class_count=class_count, pad_cells=True,
         logdir=logdir, batch_size=16, epochs=100,learning_rate=0.0005, optimizer='NAdam', verbose=True, dt_string=dt_string)

    inspect(modelpath=os.path.join(logdir, dt_string+'.h5'), X_test=X_test, y_test=y_test, mean=np.asarray([0, 0, 0]),
            size_target=size_target, pad_cells=True,
            class_id_to_name=cells['class_id_to_name'])


if __name__ == '__main__':

    annotations_path = os.path.join(get_parent_path(1), 'Data', 'Segmentations_Clinical_All')

    resistant_strain_ID = 'L13034'
    sensitive_strain_ID = 'L48480'


    experiment0 = os.path.join(get_parent_path(1), 'Data', 'Clinical_strains', 'Repeat_0_10_11_21+Repeat_1_18_11_21')
    #experiment1 = os.path.join(get_parent_path(1), 'Data', 'Clinical_strains', 'Repeat_2_03_12_21+Repeat_5_06_12_21')
    #experiment2 = os.path.join(get_parent_path(1), 'Data', 'Clinical_strains', 'Repeat_3_04_12_21')

    experiments = [experiment0]

    resistant_paths = [os.path.join(exp,'13834') for exp in experiments]
    sensitive_paths = [os.path.join(exp,'48480') for exp in experiments]


    output_path = os.path.join(get_parent_path(1), 'Data', 'Clinical_Resistant_Sensitive_Comparison')
    cond_IDs = ['CIP+ETOH']
    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)

    resistant_sensitive_comparison(output_path=output_path, cond_ID=cond_IDs, image_channels=image_channels, img_dims=img_dims,
                                   resistant_path=resistant_paths, sensitive_path=sensitive_paths, resistant_strain_ID=resistant_strain_ID,
                                   sensitive_strain_ID=sensitive_strain_ID, annotations_path=annotations_path)






