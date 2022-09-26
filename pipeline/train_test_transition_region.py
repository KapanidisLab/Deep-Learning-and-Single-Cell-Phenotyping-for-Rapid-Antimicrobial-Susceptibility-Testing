'''
Run inference on clinical samples, train on a selection and test on selection.
'''
import copy
import random

from helpers import *
from segmentation import BacConfig,predict_mrcnn_segmenter
from ProcessingPipeline import ProcessingPipeline
from classification import struct_from_file, cells_from_struct, split_cell_sets, train, inspect
from Resistant_Sensitive_Comparison import amend_class_labels, combine_cells
import seaborn as sns

def load_and_segment_image(img_path=None, segmenter=None):
    # Create an image for segmentation, fill 3 channels with NR
    img = skimage.io.imread(img_path)
    base,filename = os.path.split(img_path)

    img_NR = np.zeros(img.shape)
    img_NR[:, :, 0] = img[:, :, 0]
    img_NR[:, :, 1] = img[:, :, 0]
    img_NR[:, :, 2] = img[:, :, 0]

    # Expand to correct format
    img_NR = np.expand_dims(img_NR, axis=0)

    # Create and run segmenter
    configuration = BacConfig()
    segmentations = predict_mrcnn_segmenter(source=img_NR, mode='images', weights=segmenter,
                                            config=configuration, filenames=filename)
    # Remove all edge detections
    segmentations, removed = remove_edge_cells(segmentations)
    print('Removed {} edge cells from image.'.format(removed))

    return segmentations

def save_to_integer_encoding(masks, filename, output_path):
    '''Take in (y,x,N) binary ndarray of masks, output integer-encoded tif'''

    (y,x,N) = masks.shape
    intmask = np.ones((y,x,N), dtype='int16')
    for i in range(N): #Think of faster way later
        intmask[:,:,i] = i * intmask[:,:,i]

    masks_out = np.sum(np.multiply(masks,intmask),axis=-1)

    skimage.io.imsave(os.path.join(output_path,filename),masks_out)

def segment_and_save_masks(data_path=None, output=None, segmenter=None):
    makedir(output)

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.tif'):

                segmentation = load_and_segment_image(os.path.join(root,file), segmenter=segmenter)
                save_to_integer_encoding(masks=segmentation[0]['masks'], filename=file, output_path=output)
                print('Segmented and saved {}'.format(file))


def gather_images_by_concentration(folders=None, concentrations=None, output=None, cond_IDs=None, img_dims=None, image_channels=None,
             crop_mapping=None,registration_target=None):
    folders = [os.path.join(folder,concentration) for folder in folders for concentration in concentrations]

    makedir(output)
    pipeline = ProcessingPipeline(folders, 'NIM')
    pipeline.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
             crop_mapping=crop_mapping, output_folder=os.path.join(output,'Segregated'))
    pipeline.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=os.path.join(output,'Collected'),
                 registration_target=registration_target)

def prepare_datasets(data_sources=[], annotation_sources=[], output=None, cond_IDs=None):
    pipeline = ProcessingPipeline(data_sources[0], 'NIM')
    pipeline.collected = True

    for ann_source in annotation_sources:
        pipeline.FileOp('masks_from_integer_encoding', mask_path=ann_source, output_path=ann_source)

    pipeline.FileOp('TrainTestVal_split', data_sources=[os.path.join(source,cond_ID) for source in data_sources for cond_ID in cond_IDs],
                    annotation_sources=[os.path.join(source,'annots') for source in annotation_sources], output_folder=output, test_size=0.2,
                    validation_size=0.2, seed=42)

def select_N_cells(cells=None, N=None):
    if 'class_id_to_name' not in cells:
        raise ValueError('Class ID to name mapping not found in cell dict. Check formatting.')
    output = {}
    output['class_id_to_name'] = copy.deepcopy(cells['class_id_to_name'])
    for key in cells:
        if key == 'class_id_to_name': continue

        cell_count = len(cells[key])
        if cell_count < N: raise ValueError('Cond ID "{}" has {} cells, which is below selected threshold of {}'.format(str(key), cell_count, N))

        sample = random.sample(cells[key], k=N)
        output[key] = sample

        print('Picked {} cells from condition {}'.format(len(sample), str(key)))

    return output




if __name__ == '__main__':

    training_folders = [r'C:\Users\zagajewski\PycharmProjects\AMR\Data\23_03_22', r'C:\Users\zagajewski\PycharmProjects\AMR\Data\02_05_22']

    training_concentrations_treated = ['05','1','2','4','8','16']
    training_concentrations_untreated = ['0']

    segmenter_weights = r'C:\Users\zagajewski\Desktop\Deployment\mask_rcnn_EXP1.h5'
    classifier_weights = r'C:\Users\zagajewski\Desktop\AMR_ms_data_models\WT0CIP1_Holdout_Test\MODE - DenseNet121 BS - 16 LR - 0.0005 Holdout test.h5'

    output = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\transition_region'

    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)
    cond_IDs = ['CIP+ETOH']

    map_WT = {'colour': 'orangered', 'name': 'Untreated'}
    map_CIP = {'colour': 'dodgerblue', 'name': 'CIP'}
    mapping = {0: map_WT, 1: map_CIP}

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    makedir(output)

    #Gather images
    output_untreated = os.path.join(output,'Untreated')
    #gather_images_by_concentration(folders=training_folders, concentrations=training_concentrations_untreated,output=output_untreated, cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels, crop_mapping={'DAPI': 0, 'NR': 0}, registration_target=0)

    output_treated = os.path.join(output,'Treated')
    #gather_images_by_concentration(folders=training_folders, concentrations=training_concentrations_treated,output=output_treated, cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels, crop_mapping={'DAPI': 0, 'NR': 0}, registration_target=0)

    #Load segmenter
    print('LOADING SEGMENTER...')
    configuration = BacConfig()
    configuration.IMAGES_PER_GPU = 1
    configuration.IMAGE_RESIZE_MODE = 'pad64'  # Pad to multiples of 64
    configuration.__init__()

    segmenter = modellib.MaskRCNN(mode='inference', model_dir='../mrcnn/', config=configuration)
    segmenter.load_weights(segmenter_weights, by_name=True)
    print('DONE \n')

    #Segment masks
    output_masks_treated = os.path.join(output,'Masks_Treated')
    output_masks_untreated = os.path.join(output,'Masks_Untreated')

    assert len(cond_IDs) == 1, 'Titration data does not support multiple condition identifiers.'
    for cond_ID in cond_IDs:
        data_path_treated = os.path.join(output_treated,'Collected', cond_ID)
        #segment_and_save_masks(data_path=data_path_treated,output=output_masks_treated,segmenter=segmenter)

        data_path_untreated = os.path.join(output_untreated,'Collected', cond_ID)
        #segment_and_save_masks(data_path=data_path_untreated,output=output_masks_untreated,segmenter=segmenter)


    #Prepeare classifier training datasets
    #prepare_datasets([os.path.join(output_untreated,'Collected')],[os.path.join(output_masks_untreated)], output=os.path.join(output_untreated,'Dataset'), cond_IDs=cond_IDs)
    #prepare_datasets([os.path.join(output_treated,'Collected')],[os.path.join(output_masks_treated)], output=os.path.join(output_treated,'Dataset'), cond_IDs=cond_IDs)

    #Extract cells, and merge
    manual_struct_untreated = struct_from_file(dataset_folder=os.path.join(output_untreated,'Dataset'),class_id=1)
    manual_struct_treated = struct_from_file(dataset_folder=os.path.join(output_treated, 'Dataset'), class_id=1)

    cells_untreated = cells_from_struct(input=manual_struct_untreated, cond_IDs=cond_IDs, image_dir=os.path.join(output_untreated,'Collected'), mode='masks')
    cells_treated = cells_from_struct(input=manual_struct_treated, cond_IDs=cond_IDs, image_dir=os.path.join(output_treated,'Collected'), mode='masks')

    cells_untreated = amend_class_labels(original_label='CIP+ETOH', new_label='Untreated', new_id=0, cells=cells_untreated)
    cells_treated = amend_class_labels(original_label='CIP+ETOH', new_label='CIP', new_id=1, cells=cells_treated)

    #Balance dataset
    cells_treated = select_N_cells(cells=cells_treated, N=3500)
    cells_untreated = select_N_cells(cells=cells_untreated, N=3500)

    cells = combine_cells([cells_treated,cells_untreated])

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

    dt_string = 'DenseNet121_WTCIP_transition'

    train(mode='DenseNet121', X_train=X_train, y_train=y_train, size_target=resize_target, class_count=class_count,
          pad_cells=True,
        logdir=output, batch_size=16, epochs=100, learning_rate=0.0005, optimizer='NAdam', verbose=True,
         dt_string=dt_string)

    inspect(modelpath=os.path.join(output, dt_string + '.h5'), X_test=X_test, y_test=y_test,
            mean=np.asarray([0, 0, 0]), size_target=resize_target, pad_cells=True,
            class_id_to_name=cells['class_id_to_name'],
            colour_mapping={'Untreated': sns.light_palette((0, 75, 60), input="husl"),
                            'RIF': sns.light_palette((145, 75, 60), input="husl")})


