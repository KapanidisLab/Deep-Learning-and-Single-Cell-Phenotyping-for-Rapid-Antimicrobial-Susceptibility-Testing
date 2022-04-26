from ProcessingPipeline import ProcessingPipeline
import os
from helpers import *
import tensorflow as tf

from classification import *
from segmentation import BacConfig, predict_mrcnn_segmenter
from skimage.io import imread

from similarity import evaluate_similarity
from keras.models import load_model
from tqdm import tqdm
#Workaround for custom loss

from keras.utils.generic_utils import get_custom_objects
from triplet_loss import triplet_semihard_loss, triplet_hard_loss
from similarity import mean2mean,WT2WT_mean,CIP2WT_mean,WT2CIP_mean,CIP2CIP_mean

custom_object_handles = {"triplet_semihard_loss": triplet_semihard_loss, 'triplet_hard_loss': triplet_hard_loss, 'mean2mean':mean2mean, 'WT2WT_mean':WT2WT_mean, 'CIP2WT_mean':CIP2WT_mean, 'WT2CIP_mean':WT2CIP_mean, 'CIP2CIP_mean':CIP2CIP_mean}
get_custom_objects().update(custom_object_handles)

def segment_all(path, segmenter=None):
    cells = []
    image_count = 0
    removed_cells = 0
    # Find all images
    for root, dirs, files in os.walk(path):
        for file in tqdm(files, desc='Segmenting'):
            if file.endswith('.tif'):

                image_count += 1
                fname = os.path.join(root, file)
                img = imread(fname)

                cells_from_image, removedcount = segment_image(img,segmenter=segmenter,filename=fname)

                removed_cells += removedcount

                cells.extend(cells_from_image[0])


    print('')
    print('Segmented {} cells from {} images'.format(len(cells), image_count))
    print('Removed {} edge cells'.format(removed_cells))
    return cells

def segment_image(img=None, segmenter=None, filename=None):
    # Create an image for segmentation, fill 3 channels with NR
    img_NR = np.zeros(img.shape)
    img_NR[:, :, 0] = img[:, :, 0]
    img_NR[:, :, 1] = img[:, :, 0]
    img_NR[:, :, 2] = img[:, :, 0]

    # Expand to correct format
    img_NR = np.expand_dims(img_NR, axis=0)
    img = np.expand_dims(img, axis=0)

    # Create and run segmenter. Remove edge cells.
    configuration = BacConfig()
    segmentations = predict_mrcnn_segmenter(source=img_NR, mode='images', weights=segmenter,
                                            config=configuration, filenames=filename)

    segmentations, removedcount = remove_edge_cells(segmentations)

    cells = apply_rois_to_image(input=segmentations, mode='masks', images=img)
    return cells, removedcount

if __name__ == '__main__':

    with tf.InteractiveSession().as_default() as sess:

        data_folder = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Clinical_strains\Repeat_0_10_11_21+Repeat_1_18_11_21\17667'

        output=os.path.join(get_parent_path(1),'Data','24_01_22_Similarity_Distribution_Evaluate')
        makedir(output)

        cond_IDs = ['WT+ETOH', 'CIP+ETOH'] #first tag is untreated, second tag is treated
        image_channels = ['NR', 'DAPI']
        img_dims = (30, 684, 840)

        resize_target = (64,64,3)

        similarity_modelpath = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\02_02_22_Similarity_Full_more_shear\BS128_LR0001_SGDN_4.h5'
        segmentation_modelpath = r'C:\Users\zagajewski\Desktop\Deployment\mask_rcnn_EXP1.h5'


        # Fix pycharm console
        class PseudoTTY(object):
            def __init__(self, underlying):
                self.__underlying = underlying

            def __getattr__(self, name):
                return getattr(self.__underlying, name)

            def isatty(self):
                return True
        sys.stdout = PseudoTTY(sys.stdout)


        #Load models

        print('LOADING SEGMENTER...')
        configuration = BacConfig()
        configuration.IMAGES_PER_GPU = 1
        configuration.IMAGE_RESIZE_MODE = 'pad64'  # Pad to multiples of 64
        configuration.__init__()

        segmenter = modellib.MaskRCNN(mode='inference', model_dir='../mrcnn/', config=configuration)
        segmenter.load_weights(segmentation_modelpath, by_name=True)
        print('DONE \n')

        print('LOADING SIMILARITY ENCODER...')
        encoder = load_model(similarity_modelpath)
        print('DONE \n')


        #Separate experiments by repeat


        output_segregated = os.path.join(output, 'Segregated')
        output_collected = os.path.join(output, 'Collected')

        pipeline = ProcessingPipeline(data_folder, 'NIM')
        pipeline.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
                      crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated)
        pipeline.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected,
                         registration_target=0)

        # Segment untreated cells
        untreated_path = os.path.join(output_collected, cond_IDs[0])
        print('Segmenting untreated examples from {}'.format(untreated_path))
        cells_untreated = segment_all(untreated_path, segmenter=segmenter)

        # Segment treated cells
        treated_path = os.path.join(output_collected, cond_IDs[1])
        print('Segmenting treated examples from {}'.format(treated_path))
        cells_treated = segment_all(treated_path, segmenter=segmenter)

        # Normalize histograms to match training setup


        evaluate_similarity(model=encoder,untreated_cells=cells_untreated, treated_cells=cells_treated,target_shape=resize_target)


