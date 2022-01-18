# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 19:33:56 2020

@author: Aleksander Zagajewski

This is an internal handle to integrate mrcnn functionality with the rest of the pipeline.
"""

from mrcnn import config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize as visualize

from mrcnn.model import log

from helpers import *

# Import all the necessary libraries
import os
import sys
import skimage.io  # Used for imshow function
import skimage.transform  # Used for resize function
import numpy as np
import keras
import tensorflow as tf
import sklearn
import warnings
import sklearn.metrics
import time

import copy

import concurrent.futures
import multiprocessing
import itertools
import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------MODEL CONFIGURATION CLASS---------------------------------------------------
class BacConfig(config.Config):
    #Inherit from base class, change settings where necessary

    NAME = 'BAC'

    LEARNING_RATE = 0.003
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1  # BG and Cell

    STEPS_PER_EPOCH = None # This is calculated later

    MEAN_PIXEL = np.array([0, 0, 0])

    IMAGE_CHANNEL_COUNT = 3

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    RPN_ANCHOR_SCALES = (8,16,32,64,128)  # Anchor scales decreased to match size of bacteria better. 5 values to match backbone strides.
    RPN_NMS_THRESHOLD = 0.9
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    TRAIN_ROIS_PER_IMAGE = 200
    IMAGE_MIN_SCALE = 1

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.2

    def set(self, attribute, value):
        setattr(self, attribute, value)
        return getattr(self, attribute)

#-----------------------------------------------------------------------------------------------------------------------


#----------------------------------------------MODEL DATASET CLASS------------------------------------------------------
class BacDataset(utils.Dataset):

    def load_dataset(self, dataset_dir):

        '''
        Loads requisite data and annotations in standard format from rest of pipeline.
        Class scheme supports easy and arbitrary distribution of data sources between classes.

        :param dataset_dir (list of strings) - list of data sources. Data sou
        :param class_scheme:
        :return:
        '''

        #Check that dataset_dir and class_scheme are both lists


        import re

        # Define bacteria class
        self.add_class('dataset', 1, 'Cell')

        # Define data locations
        imdir = os.path.join(dataset_dir, 'images')
        andir = os.path.join(dataset_dir, 'annots')

        # find images
        for idx, filename in enumerate(os.listdir(imdir)):
            filename_delimited = re.split('[_.]', filename)
            image_id = idx

            img_path = os.path.join(imdir, filename)
            an_path = os.path.join(andir, os.path.splitext(filename)[0])  # Path to folder with image annotations, remove filename extension

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=an_path)

    def load_mask(self, image_id):  # here the image_id is integer index loaded by add_image

        import numpy as np

        # load image_id stored by load_dataset
        info = self.image_info[image_id]
        path = info['annotation']
        image_path = info['path']

        img = self.load_image(image_id)
        (height, width, _) = img.shape

        labelcount = len(os.listdir(path))

        masks = np.zeros([height, width, labelcount], dtype='uint8')  # initialize array for masks
        class_ids = list()  # initialise array for class ids of masks

        for count, filename in enumerate(os.listdir(path)):
            masks[:, :, count] = skimage.io.imread(os.path.join(path, filename))
            class_ids.append(self.class_names.index('Cell'))

        masks[masks >= 244] = 1  # Binarize mask

        return masks, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
#-----------------------------------------------------------------------------------------------------------------------


def train_mrcnn_segmenter(**kwargs):

    train_folder = kwargs.get('train_folder', False)
    validation_folder = kwargs.get('validation_folder', False)
    config = kwargs.get('configuration', False)
    augmentation = kwargs.get('augmentation', False)
    weights = kwargs.get('weights', False)
    output_folder = kwargs.get('output_folder', False)

    if not all([train_folder, validation_folder, config, augmentation, weights, output_folder]):  # Verify input
        raise TypeError

    configuration = copy.deepcopy(config) #Local copy of input config to avoid modifying outside local scope

    print('Python       :', sys.version.split('\n')[0])
    print('Numpy        :', np.__version__)
    print('Skimage      :', skimage.__version__)
    print('Scikit-learn :', sklearn.__version__)
    print('Keras        :', keras.__version__)
    print('Tensorflow   :', tf.__version__)
    print('')

    sys.stdout.flush()

    import imgaug.augmenters as iaa  # import augmentation library

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0 #Select device to train on

    # Load sets
    train_set = BacDataset()
    train_set.load_dataset(train_folder)
    train_set.prepare()

    val_set = BacDataset()
    val_set.load_dataset(validation_folder)
    val_set.prepare()

    print('----------------------------------------------------------')
    print('           TRAINING THE MODEL ON TRAINING SET')
    print('----------------------------------------------------------')
    print('Train: ', len(train_set.image_ids))
    print('Validation:', len(val_set.image_ids) )
    print('Weights: ', weights)
    print('Class names: ', train_set.class_names)
    print('Class IDs', train_set.class_ids)
    print('----------------------------------------------------------')


    assert train_set.class_names == val_set.class_names, 'Validation and Training need the same class distribution'

    # Define augmentation scheme

    import imgaug.augmenters as iaa  # import augmentation library
    augmentation = iaa.Sequential(augmentation)  # Execute in sequence from 1st to last

    # Define and train model
    print('Preparing to train...\n')

    warnings.filterwarnings('ignore', '', iaa.base.SuspiciousSingleImageShapeWarning, '',
                            0)  # Filter warnings from imgaug

    configuration.STEPS_PER_EPOCH = int(np.round(len(train_set.image_ids) / config.BATCH_SIZE))  # Set 1 epoch = 1 whole pass
    configuration.VALIDATION_STEPS = int(np.round(len(val_set.image_ids) / config.BATCH_SIZE)) # Iterate through entire validation set
    configuration.display()

    with tf.device(DEVICE):
        trmodel = modellib.MaskRCNN(mode='training', model_dir=output_folder, config=configuration)
        trmodel.load_weights(weights, by_name=True,
                             exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

        trmodel.train(train_set, val_set, learning_rate=configuration.LEARNING_RATE, epochs=50, layers='heads',
                      augmentation=augmentation)
        trmodel.train(train_set, val_set, learning_rate=configuration.LEARNING_RATE, epochs=100, layers='all',
                      augmentation=augmentation)
        trmodel.train(train_set, val_set, learning_rate=configuration.LEARNING_RATE / 10, epochs=150, layers='heads',
                      augmentation=augmentation)
        trmodel.train(train_set, val_set, learning_rate=configuration.LEARNING_RATE / 10, epochs=200, layers='all',
                      augmentation=augmentation)

def predict_mrcnn_segmenter(source = None, mode = None, **kwargs):

    #Use this when no corresponding gt labels are provided

    config = kwargs.get('config', False)
    weights = kwargs.get('weights', False)

    if not all([config, weights]):  # Verify input
        raise TypeError

    assert mode in ['images', 'dataset'], 'Predictions either from dataset object or raw images'

    if mode == 'dataset':

        if type(source) == str:
            assert os.path.isdir(source), 'In dataset mode, the image source must be a path to test directory or dataset object.'
            test_set = BacDataset()
            test_set.load_dataset(source)
            test_set.prepare()

        else:
            assert isinstance(source, BacDataset), 'In dataset mode, the image source must be a path to test directory or dataset object.'
            test_set = source

        image_count = len(test_set.image_ids)

    elif mode == 'images':

        assert isinstance(source, np.ndarray),' In images mode, the image source must be an ndarray.'
        assert len(source.shape) == 4, 'Images must be in format (n_samples,x,y,ch)'
        (N,x,y,ch) = source.shape #Get source info
        image_count = N

    output = []

    configuration = copy.deepcopy(config)  # Local copy of config to avoid modifying in outer scope

    #TODO The model.detect() can process in larger batch sizes. Consider refactoring for increased speed.

    configuration.IMAGES_PER_GPU = 1
    configuration.IMAGE_RESIZE_MODE = 'pad64' #Pad to multiples of 64
    configuration.__init__()

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0 #Select device to execute on

    with tf.device(DEVICE):

        #Allow option to load model outside of function call
        if isinstance(weights, modellib.MaskRCNN):
            model = weights
        else:

            model = modellib.MaskRCNN(mode='inference', model_dir='../mrcnn/', config=configuration)
            model.load_weights(weights, by_name=True)

        for i in range(image_count):

            if mode == 'dataset':
                image_id = test_set.image_ids[i]
                image = test_set.load_image(image_id)

                assert test_set.image_info[i]['id'] == image_id #Ensure we got the correct image
                filename = os.path.split(test_set.image_info[i]['path'])[1] #Get filename


            elif mode == 'images':
                image = source[i,:,:,:]
                image_id = i
                filenames = kwargs.get('filenames', None) #Optional parameter if in images mode
                filename = filenames[i]

            results = model.detect(np.expand_dims(image, 0), verbose=0)[0] #Run detection routine

            results['filename'] = filename #Store filename, or none
            results['image_id'] = image_id
            output.append(results) #Store

    return output



def inspect_mrcnn_segmenter(ids = None, **kwargs):

    #ids - list or int, select specific image ids for inspection. If None, loop through entire dataset

    test_folder = kwargs.get('test_folder', False)
    config = kwargs.get('configuration', False)
    weights = kwargs.get('weights', False)

    if not all([test_folder, config, weights]):  # Verify input
        raise TypeError

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0 #Select device to train on

    configuration = copy.deepcopy(config) #Local copy of config to avoid modifying in outer scope

    test_set = BacDataset()
    test_set.load_dataset(test_folder)
    test_set.prepare()


    print('----------------------------------------------------------')
    print('            INSPECTING THE MODEL ON TEST SET')
    print('----------------------------------------------------------')
    print('Test: ', len(test_set.image_ids))
    print('Weights: ' + weights)
    print('Class names: ', test_set.class_names)
    print('Class IDs', test_set.class_ids)
    print('----------------------------------------------------------')

    configuration.IMAGES_PER_GPU = 1  # Process one image at a time
    configuration.IMAGE_RESIZE_MODE = "square"  # Resize images to make the figures nice and pretty
    configuration.IMAGE_MIN_DIM = 800
    configuration.IMAGE_MAX_DIM = 1024
    configuration.__init__()  # Reinitialise to update values of computed parameters

    #TODO model.detect can support a batch>1, no need to feed ids one at a time. Consider rewriting.

    # define the model
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='inference', model_dir='../mrcnn/', config=configuration)
        model.load_weights(weights, by_name=True)

        if ids is None:
            image_ids = test_set.image_ids  #Loop through entire set if not supplied
        elif not hasattr(ids, '__len__'):   #Expand to list if single value
            image_ids = [ids]
            assert len(image_ids) == 1
        else:                               #Pass iterable if given one
            image_ids = ids


        for image_id in image_ids: #Iterate through all images

            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(test_set, configuration, image_id, use_mini_mask=False)

            # Run object detection

            results = model.detect(np.expand_dims(image, 0), verbose=0)

            # Display results
            r = results[0]
            print('')
            print('---GT---')
            log("gt_class_id", gt_class_id)
            log("gt_bbox", gt_bbox)
            log("gt_mask", gt_mask)

            print('')
            print('---RESULTS---')
            log('ROIs', r['rois'])
            log('Class_ID', r['class_ids'])
            log('Scores', r['scores'])
            log('Masks', r['masks'])

            mAPs = []

            # Compute mAP over range 0.5 to 0.95 and print it, for this image
            mAP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                                   r['rois'], r['class_ids'], r['scores'], r['masks'],
                                   verbose=1)

            mAPs.append(mAP) #Store mAP from that image

            image_ubyte = skimage.img_as_ubyte(image)  # Convert to 8bit

            title = configuration.NAME + str('_IMAGE_ID_')+str(image_id)+str('_AP_')+str(np.around(AP,2))

            visualize.display_differences(
                image_ubyte,
                gt_bbox, gt_class_id, gt_mask,
                r['rois'], r['class_ids'], r['scores'], r['masks'],
                test_set.class_names,
                show_box=False, show_mask=False,
                iou_threshold=0, score_threshold=0, ax=get_ax(), title = title)
            plt.show()

            compute_pixel_metrics(test_set, [image_id], results)  # Compute pixel confusion mats

        print('AP:0.5-0.9 per image average: ' + str( np.around(mAPs,2)))



def inspect_dataset(**kwargs):

    dataset_folder = kwargs.get('dataset_folder', False)
    if not dataset_folder:
        raise TypeError

    dataset = BacDataset()
    dataset.load_dataset(dataset_folder)
    dataset.prepare()



    print('----------------------------------------------------------')
    print('                   INSPECTING DATASET')
    print('----------------------------------------------------------')
    print('Dataset: ', len(dataset.image_ids))
    print('Class names: ', dataset.class_names)
    print('Class IDs', dataset.class_ids)
    print('----------------------------------------------------------')

    # Display 1 random image
    image_ids = np.random.choice(dataset.image_ids, 1)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(skimage.img_as_ubyte(image), mask, class_ids, dataset.class_names, limit=1)

    # Collect image stats from the whole dataset

    # Loop through the dataset and compute stats over multiple threads
    # This might take a few minutes
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as e:
        stats = list(e.map(image_stats, dataset.image_ids, itertools.repeat(dataset)))
    t_total = time.time() - t_start
    print("Total time: {:.1f} seconds".format(t_total))

    # Image stats

    image_shape = np.array([s['shape'] for s in stats])
    image_color = np.array([s['color'] for s in stats])
    print("Image Count: ", image_shape.shape[0])
    print("Height  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
        np.mean(image_shape[:, 0]), np.median(image_shape[:, 0]),
        np.min(image_shape[:, 0]), np.max(image_shape[:, 0])))
    print("Width   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
        np.mean(image_shape[:, 1]), np.median(image_shape[:, 1]),
        np.min(image_shape[:, 1]), np.max(image_shape[:, 1])))
    print("Color   mean (RGB): {:.2f} {:.2f} {:.2f}".format(*np.mean(image_color, axis=0)))

    # Histograms
    fig, ax = plt.subplots(3, 1)
    fig.suptitle('Image shape statistics', y=1)
    fig.tight_layout()

    ax[0].set_title("Height")
    ax[0].set_xlabel('Pixels')
    ax[0].set_ylabel('Freq Den')
    _ = ax[0].hist(image_shape[:, 0], bins=20)

    ax[1].set_title("Width")
    ax[1].set_xlabel('Pixels')
    ax[1].set_ylabel('Freq Den')
    _ = ax[1].hist(image_shape[:, 1], bins=20)

    ax[2].set_title("Height & Width")
    ax[2].set_xlabel('Pixels Width')
    ax[2].set_ylabel('Pixels Height')
    _ = ax[2].hist2d(image_shape[:, 1], image_shape[:, 0], bins=10, cmap="Blues")

    #fig.subplots_adjust(top=0.65)

    plt.show()

    # Objects per image stats

    # Segment by image area
    image_area_bins = [256 ** 2, 600 ** 2, 1300 ** 2]

    print("Objects/Image")
    fig, ax = plt.subplots(len(image_area_bins), 1)
    fig.suptitle('Distribution of objects per image', y=1)
    fig.tight_layout()

    area_threshold = 0
    for i, image_area in enumerate(image_area_bins):
        objects_per_image = np.array([len(s['bbox'])
                                      for s in stats
                                      if area_threshold < (s['shape'][0] * s['shape'][1]) <= image_area])
        area_threshold = image_area
        if len(objects_per_image) == 0:
            print("Image area <= {:4}**2: None".format(np.sqrt(image_area)))
            continue
        print("Image area <= {:4.0f}**2:  mean: {:.1f}  median: {:.1f}  min: {:.1f}  max: {:.1f}".format(
            np.sqrt(image_area), objects_per_image.mean(), np.median(objects_per_image),
            objects_per_image.min(), objects_per_image.max()))
        ax[i].set_title("Image Area <= {:4}**2".format(np.sqrt(image_area)))
        ax[i].set_xlabel('Objects per image')
        ax[i].set_ylabel('Freq Den')
        _ = ax[i].hist(objects_per_image, bins=10)

    #fig.subplots_adjust(top=0.65)
    plt.show()

    # Object size stats

    fig, ax = plt.subplots(len(image_area_bins),1)
    fig.suptitle('Object size statistics', y=1)
    fig.tight_layout()

    area_threshold = 0
    for i, image_area in enumerate(image_area_bins):
        object_shape = np.array([
            b
            for s in stats if area_threshold < (s['shape'][0] * s['shape'][1]) <= image_area
            for b in s['bbox']])

        try:

            object_area = object_shape[:, 0] * object_shape[:, 1]
            area_threshold = image_area

            print("\nImage Area <= {:.0f}**2".format(np.sqrt(image_area)))
            print("  Total Objects: ", object_shape.shape[0])
            print("  Object Height. mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
                np.mean(object_shape[:, 0]), np.median(object_shape[:, 0]),
                np.min(object_shape[:, 0]), np.max(object_shape[:, 0])))
            print("  Object Width.  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
                np.mean(object_shape[:, 1]), np.median(object_shape[:, 1]),
                np.min(object_shape[:, 1]), np.max(object_shape[:, 1])))
            print("  Object Area.   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
                np.mean(object_area), np.median(object_area),
                np.min(object_area), np.max(object_area)))

            # Show 2D histogram
            _ = ax[i].hist2d(object_shape[:, 1], object_shape[:, 0], bins=20, cmap="Blues")
            ax[i].set_xlabel('Object width (pix)')
            ax[i].set_ylabel('Object height (pix)')
            ax[i].set_title("Image Area <= {:4}**2".format(np.sqrt(image_area)))

        except IndexError as error:
            print("\nImage Area <= {:.0f}**2".format(np.sqrt(image_area)), '- No Objects Found!')

    #fig.subplots_adjust(top=0.65)
    plt.show()

def inspect_augmentation(**kwargs):

    dataset_folder = kwargs.get('dataset_folder', False)
    config = kwargs.get('configuration', False)
    augmentation = kwargs.get('augmentation', False)

    if not all([dataset_folder, config, augmentation]):  # Verify input
        raise TypeError

    configuration = copy.deepcopy(config) #Local copy of configuration to avoid modifying input in outer scope

    dataset = BacDataset()
    dataset.load_dataset(dataset_folder)
    dataset.prepare()

    print('----------------------------------------------------------')
    print('           INSPECTING AUGMENTATION ON DATASET')
    print('----------------------------------------------------------')
    print('Dataset: ', len(dataset.image_ids))
    print('Class names: ', dataset.class_names)
    print('Class IDs', dataset.class_ids)
    print('----------------------------------------------------------')

    import imgaug.augmenters as iaa  # import augmentation library

    augmentation = iaa.Sequential(augmentation)  # Execute in sequence from 1st to last

    image_id = np.random.choice(dataset.image_ids, 1)[0] # Select random image to show augs on

    limit = 4
    ax = get_ax(rows=2, cols=2)
    for i in range(limit):
        image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
            dataset, configuration, image_id, use_mini_mask=False, augment=False, augmentation=augmentation)

        #To better visualise zero-padding, set padded zeros to a grey level
        image[image == 0] = 25000

        #Visualise
        visualize.display_instances(skimage.img_as_ubyte(image), bbox, mask, class_ids,
                                    dataset.class_names, ax=ax[i // 2, i % 2],
                                    show_mask=False, show_bbox=False, title='Augmentation example {}'.format(i))

    plt.show()


def inspect_segmenter_stepwise(**kwargs):

    train_folder = kwargs.get('train_folder', False)
    test_folder = kwargs.get('test_folder', False)
    config = kwargs.get('configuration', False)
    weights = kwargs.get('weights', False)

    if not all([train_folder, test_folder, config, weights]):  # Verify input
        raise TypeError

    configuration = copy.deepcopy(config) #Local copy to avoid modifying input in outer scope

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0 #Select device to train on

    test_set = BacDataset()
    test_set.load_dataset(test_folder)
    test_set.prepare()

    train_set = BacDataset()
    train_set.load_dataset(train_folder)
    train_set.prepare()

    print('----------------------------------------------------------')
    print('              STEPWISE MODEL INSPECTION')
    print('----------------------------------------------------------')
    print('Train: ', len(train_set.image_ids))
    print('Test: ', len(train_set.image_ids))
    print('Weights: ', weights)
    print('Class names: ', train_set.class_names)
    print('Class IDs', train_set.class_ids)
    print('----------------------------------------------------------')

    configuration.IMAGES_PER_GPU = 1  # Process one image at a time
    configuration.IMAGE_RESIZE_MODE = "square"  # Resize images to make the figures nice and pretty
    configuration.IMAGE_MIN_DIM = 800
    configuration.IMAGE_MAX_DIM = 1024
    configuration.__init__()  # Reinitialise to update values of computed parameters

    # define the model
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='inference', model_dir='../mrcnn/', config=configuration)
        model.load_weights(weights, by_name=True)

        image_id_train = np.random.choice(train_set.image_ids, 1)[0]  # Select random image to inspect from train
        image_id_test = np.random.choice(test_set.image_ids, 1)[0]  # Select random image to inspect from test

        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(test_set, configuration, image_id_test, use_mini_mask=False)

        image_ubyte = skimage.img_as_ubyte(image)  # Convert to 8bit for visualisation

        # ************************************************* (1) RPN ********************************************************
        #
        # The Region Proposal Network (RPN) runs a lightweight binary classifier on a lot of boxes (anchors) over the image
        # and returns object/no-object scores. Anchors with high objectness score (positive anchors) are passed to the stage
        # two to be classified.
        #
        # Often, even positive anchors don't cover objects fully. So the RPN also regresses a refinement (a delta in
        # location and size) to be applied to the anchors to shift it and resize it a bit to the correct boundaries of
        # the object.

        # ============================================= (A) RPN targets ====================================================

        # The RPN targets are the training values for the RPN. To generate the targets, we start with a grid of anchors
        # that cover the full image at different scales, and then we compute the IoU of the anchors with ground truth
        # object. Positive anchors are those that have an IoU >= 0.7 with any ground truth object, and negative anchors
        # are those that don't cover any object by more than 0.3 IoU. Anchors in between (i.e. cover an object by IoU >=
        # 0.3 but < 0.7) are considered neutral and excluded from training.
        #
        # To train the RPN regressor, we also compute the shift and resizing needed to make the anchor cover the ground
        # truth object completely.

        print('*********************STEP BY STEP********************')
        print('')
        print('===================== RPN TRAIN =====================')
        print('')

        # Get anchors and convert to pixel coordinates
        anchors = model.get_anchors(image.shape)
        anchors = utils.denorm_boxes(anchors, image.shape[:2])
        log("anchors", anchors)

        # Generate RPN trainig targets
        # target_rpn_match is 1 for positive anchors, -1 for negative anchors
        # and 0 for neutral anchors.
        target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
            image.shape, anchors, gt_class_id, gt_bbox, model.config)
        log("target_rpn_match", target_rpn_match)
        log("target_rpn_bbox", target_rpn_bbox)

        positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
        negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
        neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
        positive_anchors = anchors[positive_anchor_ix]
        negative_anchors = anchors[negative_anchor_ix]
        neutral_anchors = anchors[neutral_anchor_ix]
        log("positive_anchors", positive_anchors)
        log("negative_anchors", negative_anchors)
        log("neutral anchors", neutral_anchors)

        # Apply refinement deltas to positive anchors
        refined_anchors = utils.apply_box_deltas(
            positive_anchors,
            target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
        log("refined_anchors", refined_anchors, )

        # Display positive anchors before refinement (dotted) and
        # after refinement (solid).
        visualize.draw_boxes(
            image_ubyte, ax=get_ax(),
            boxes=positive_anchors,
            refined_boxes=refined_anchors, title='RPN training - Positive anchors')
        plt.show()

        # Inspect negative anchors'
        visualize.draw_boxes(
            image_ubyte, ax=get_ax(),
            boxes=negative_anchors, title='RPN training - Negative anchors')
        plt.show()

        print('')
        print('==================== RPN PREDICTION ====================')
        print('')

        # ============================================= (B) RPN predictions ===============================================
        # Run RPN sub-graph
        pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

        # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
        if nms_node is None:
            nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
        if nms_node is None:  # TF 1.9-1.10
            nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

        rpn = model.run_graph(image[np.newaxis], [
            ("rpn_class", model.keras_model.get_layer("rpn_class").output),
            ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
            ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
            ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
            ("post_nms_anchor_ix", nms_node),
            ("proposals", model.keras_model.get_layer("ROI").output),
        ], image_metas=image_meta[np.newaxis])

        # Plot top 100 predictions at various stages
        limit = 100
        sorted_anchor_ids = np.argsort(rpn['rpn_class'][:, :, 1].flatten())[::-1]
        visualize.draw_boxes(image_ubyte, boxes=anchors[sorted_anchor_ids[:limit]], ax=get_ax(),
                             title='RPN predictions, top '
                                   '100 anchors. Before '
                                   'refinement. STEP 1.')
        plt.show()

        pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
        refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
        refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
        visualize.draw_boxes(image_ubyte, refined_boxes=refined_anchors_clipped[:limit], ax=get_ax(),
                             title='RPN predictions, '
                                   'top 100 anchors. '
                                   'After refinement '
                                   'and edge clipping. STEP 2')
        plt.show()

        ixs = rpn["post_nms_anchor_ix"][:limit]
        visualize.draw_boxes(image_ubyte, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax(),
                             title='RPN predictions, top '
                                   '100 anchors. After '
                                   'NMS. STEP 3')
        plt.show()

        # ************************************************* (2) CLASSIFIER *************************************************
        # Run a classfier on proposals

        print('')
        print('==================== CLASSIFIER ====================')
        print('')

        mrcnn = model.run_graph([image], [
            ("proposals", model.keras_model.get_layer("ROI").output),
            ("probs", model.keras_model.get_layer("mrcnn_class").output),
            ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
            ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ])

        # Proposals are in normalized coordinates
        proposals = mrcnn["proposals"][0]

        # Class ID, score, and mask per proposal
        roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
        roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
        roi_class_names = np.array(test_set.class_names)[roi_class_ids]
        roi_positive_ixs = np.where(roi_class_ids > 0)[0]

        # How many ROIs vs empty rows?
        print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
        print("{} Positive ROIs".format(len(roi_positive_ixs)))

        # Class counts
        print(list(zip(*np.unique(roi_class_names, return_counts=True))))

        # Display a random sample of proposals.
        # Proposals classified as background are dotted, and
        # the rest show their class and confidence score.
        limit = 200
        ixs = np.random.randint(0, proposals.shape[0], limit)
        captions = ["{} {:.3f}".format(test_set.class_names[c], s) if c > 0 else ""
                    for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
        visualize.draw_boxes(
            image_ubyte,
            boxes=utils.denorm_boxes(proposals[ixs], image.shape[:2]),
            visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
            captions=captions, title="ROIs Before Refinement. STEP 4.",
            ax=get_ax())

        plt.show()

        # Apply BBox refinement

        # Class-specific bounding box shifts.
        roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
        log("roi_bbox_specific", roi_bbox_specific)

        # Apply bounding box transformations
        # Shape: [N, (y1, x1, y2, x2)]
        refined_proposals = utils.apply_box_deltas(
            proposals, roi_bbox_specific * configuration.BBOX_STD_DEV)
        log("refined_proposals", refined_proposals)

        limit = 50  # Display 10 random positive proposals
        ids = np.random.randint(0, len(roi_positive_ixs), limit)

        captions = ["{} {:.3f}".format(test_set.class_names[c], s) if c > 0 else ""
                    for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
        visualize.draw_boxes(
            image_ubyte, ax=get_ax(),
            boxes=utils.denorm_boxes(proposals[roi_positive_ixs][ids], image.shape[:2]),
            refined_boxes=utils.denorm_boxes(refined_proposals[roi_positive_ixs][ids], image.shape[:2]),
            visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
            captions=captions, title="ROIs After Refinement. 50 random positives. STEP 5.")

        plt.show()

        # Filter low-confidence detections

        # Remove boxes classified as background
        keep = np.where(roi_class_ids > 0)[0]
        print("Keep {} detections:\n{}".format(keep.shape[0], keep))

        # Remove low confidence detections
        keep = np.intersect1d(keep, np.where(roi_scores >= configuration.DETECTION_MIN_CONFIDENCE)[0])
        print("Remove boxes below {} confidence. Keep {}:\n{}".format(
            configuration.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))

        # Per class non-max suppression

        # Apply per-class non-max suppression
        pre_nms_boxes = refined_proposals[keep]
        pre_nms_scores = roi_scores[keep]
        pre_nms_class_ids = roi_class_ids[keep]

        nms_keep = []
        for class_id in np.unique(pre_nms_class_ids):
            # Pick detections of this class
            ixs = np.where(pre_nms_class_ids == class_id)[0]
            # Apply NMS
            class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
                                                   pre_nms_scores[ixs],
                                                   configuration.DETECTION_NMS_THRESHOLD)
            # Map indicies
            class_keep = keep[ixs[class_keep]]
            nms_keep = np.union1d(nms_keep, class_keep)
            print("{:22}: {} -> {}".format(test_set.class_names[class_id][:20],
                                           keep[ixs], class_keep))

        keep = np.intersect1d(keep, nms_keep).astype(np.int32)
        print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))

        # Show final detections
        ixs = np.arange(len(keep))  # Display all
        # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
        captions = ["{} {:.3f}".format(test_set.class_names[c], s) if c > 0 else ""
                    for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
        visualize.draw_boxes(
            image_ubyte,
            boxes=utils.denorm_boxes(proposals[keep][ixs], image.shape[:2]),
            refined_boxes=utils.denorm_boxes(refined_proposals[keep][ixs], image.shape[:2]),
            visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
            captions=captions, title="Detections after NMS. STEP 6.",
            ax=get_ax())

        plt.show()

        print('ALL DONE!')

def optimise_mrcnn_segmenter(mode = None, arg_names = None, arg_values = None, **kwargs):

    #Optimise parameters and train for each permutation
    # arg_names - list of strings, where each entry corresponds to a parameter to optimise
    # arg_values - list of lists of parameter values to iterate through

    assert mode in ['training', 'inference'], 'Optimise either training or inference'

    #Get common parameters
    weights = kwargs.get('weights', False)
    config = kwargs.get('configuration', False)

    if not all([weights, config]):
        raise TypeError

    #Get specific paramters
    if mode == 'training':
        train_folder = kwargs.get('train_folder', False)
        validation_folder = kwargs.get('validation_folder', False)
        augmentation = kwargs.get('augmentation', False)
        output_folder = kwargs.get('output_folder', False)

        if not all([train_folder, validation_folder, augmentation, output_folder]):  # Verify input
            raise TypeError

    elif mode == 'inference':
        test_folder = kwargs.get('test_folder', False)
        ids = kwargs.get('ids', False)
        if not all([test_folder, ids]):
            raise TypeError


    configuration = copy.deepcopy(config) #Local copy to avoid out of scope modification of input

    #Check supplied parameters
    argcheck = map(hasattr, itertools.repeat(configuration), arg_names) #Check existance of arguments
    assert all(list(argcheck)), 'Argument names must match configuration attribute names'
    assert len(arg_names) == len(arg_values), 'Argument names must have a corresponding value range'

    #Run over each permutation
    for permutation in itertools.product(*arg_values): #Permute all parameters

        newpars = list(map(configuration.set ,arg_names, permutation)) #Update config parameters with permutation
        assert newpars == list(permutation) #Confirm successful update

        configuration.NAME = 'OPTIMISATION_'+str(mode).upper() +'_PARAMETERS_'+str(arg_names)+'_VALUES_'+str(permutation) #Update name
        configuration.__init__() #Update values of computed parameters

        if mode == 'training':
            kwargs = {'train_folder':train_folder, 'validation_folder':validation_folder, 'configuration':configuration, 'augmentation':augmentation, 'weights':weights, 'output_folder':output_folder}
            p = multiprocessing.Process(target=train_mrcnn_segmenter, kwargs = kwargs )
            p.start()
            p.join()

        elif mode =='inference':
            kwargs = {'test_folder':test_folder, 'configuration':configuration, 'weights':weights, 'ids': ids}
            p = multiprocessing.Process(target=inspect_mrcnn_segmenter, kwargs = kwargs )
            p.start()
            p.join()

        else:
            raise TypeError

def evaluate_coco_metrics(dataset_folder= None, config=None, weights=None, eval_type="bbox", plot=True):
    #Runs official coco metrics calculator. Requires coco.py with alternative data loader to work with dataset object

    assert dataset_folder != None and config != None and weights != None

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as maskUtils

    def build_coco_results(image_id, rois, class_ids, scores, masks):

        # If no results, return an empty list
        if rois is None:
            return []

        results = []

        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
        return results

    dataset = BacDataset()
    dataset.load_dataset(dataset_folder)
    dataset.prepare()

    configuration = copy.deepcopy(config)
    configuration.DETECTION_MIN_CONFIDENCE = 0  # Set no artificial confidence threshold.

    COCOgt = COCO(dataset)

    results = predict_mrcnn_segmenter(source=dataset, mode='dataset', config=configuration, weights=weights)

    coco_results = []
    for result in results:
        image_results = build_coco_results(result["image_id"],
                                           result["rois"], result["class_ids"],
                                           result["scores"],
                                           result["masks"].astype(np.uint8))
        coco_results.extend(image_results)
    COCOres = COCOgt.loadRes(coco_results)  # Create COCO object from our transformed results

    # Call evaluation
    cocoEval = COCOeval(COCOgt, COCOres, eval_type)
    cocoEval.params.imgIds = dataset.image_ids  # Feed all image ids for evaluation
    cocoEval.params.maxDets = [10, 100, 1000]  # Increase max det thresholds to match dataset
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Plot PR curves if asked

    # Plot over all areas, average across categories
    if plot:
        # Find IoU0.5 and IoU0.75 and IoU0.9 parameter index
        iouThresholds = cocoEval.eval['params'].iouThrs
        id50 = np.where((np.around(iouThresholds, 1) == 0.5))[0][
            0]  # Round to 1 dp to elimiate machine precision comparison problems
        id75 = np.where((np.around(iouThresholds, 2) == 0.75))[0][0]
        id90 = np.where((np.around(iouThresholds, 1) == 0.9))[0][0]

        # Get recall therholds
        recThresholds = cocoEval.eval['params'].recThrs

        # TODO Rewrite these using a single index with 3 values

        # Get precisions at IoUs from index
        precisions = cocoEval.eval['precision']

        pr_50 = precisions[id50, :, :, 0, 2]  # data for IoU@0.5, all areas and max detections, all categories
        pr_75 = precisions[id75, :, :, 0, 2]  # data for IoU@0.75
        pr_90 = precisions[id90, :, :, 0, 2]  # data for IoU@0.90

        pr_50 = np.mean(pr_50, 1)  # Average all over categories
        pr_75 = np.mean(pr_75, 1)
        pr_90 = np.mean(pr_90, 1)

        # Get confidence threshold scores corresponding
        scores = cocoEval.eval['scores']

        scr_50 = scores[id50, :, :, 0, 2]
        scr_75 = scores[id75, :, :, 0, 2]
        scr_90 = scores[id90, :, :, 0, 2]

        scr_50 = np.mean(scr_50, 1)  # Average all over categories
        scr_75 = np.mean(scr_75, 1)
        scr_90 = np.mean(scr_90, 1)

        # Plot

        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle('Precision-Recall curve: COCO 2017 standard', y=1.05)

        axs[0].set_title('Precision-Recall: IOU@0.5, 0.75, 0.9')
        axs[0].plot(recThresholds, pr_50, color='blue')
        axs[0].plot(recThresholds, pr_75, color='green')
        axs[0].plot(recThresholds, pr_90, color='red')
        axs[0].legend(['IoU @ 0.5', 'IoU @ 0.75', 'IoU @ 0.9'])
        axs[0].xaxis.set_label_text('Recall')
        axs[0].yaxis.set_label_text('Precision')

        axs[1].set_title('Confidence at Recall thresholds: IOU@0.5, 0.75, 0.9')
        axs[1].plot(recThresholds, scr_50, color='blue')
        axs[1].plot(recThresholds, scr_75, color='green')
        axs[1].plot(recThresholds, scr_90, color='red')
        axs[1].legend(['IoU @ 0.5', 'IoU @ 0.75', 'IoU @ 0.9'])
        axs[1].xaxis.set_label_text('Recall')
        axs[1].yaxis.set_label_text('Confidence threshold')

        print('Preparing graphs...')
        plt.show()
        print('Graphs done!')

def segmentations_to_OUFTI(input_struct, filenames, setfile, output_folder):
    #Takes output struct from predict_mrcnn_segmenter and writes them in OUFTI compatible mat files. Requiers filenames
    #for file writing, and a path to setfile, which contains oufti settings to embed.

    def calculate_outline(cell_obj):
        """
        Plot the outline of the cell based on the current coordinate system.
        The outline consists of two semicircles and two offset lines to the central parabola.[1]_[2]_
        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ax.plot().
        Returns
        -------
        x,y : outline points
        p1 : (x,y) tuple, index of first pole
        p2 : (x,y) tuple, index of second pole
        """

        # Parametric plotting of offset line
        # http://cagd.cs.byu.edu/~557/text/ch8.pdf
        #
        # Analysis of the offset to a parabola
        # https://doi-org.proxy-ub.rug.nl/10.1016/0167-8396(94)00038-T

        numpoints = 50  # vertices for linear sections of cell
        numpoints_circle = 40  # vertices for both semicircles

        t = np.linspace(cell_obj.coords.xl, cell_obj.coords.xr, num=numpoints)
        a0, a1, a2 = cell_obj.coords.coeff

        x_top = t + cell_obj.coords.r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
        y_top = a0 + a1 * t + a2 * (t ** 2) - cell_obj.coords.r * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))

        x_bot = t + - cell_obj.coords.r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
        y_bot = a0 + a1 * t + a2 * (t ** 2) + cell_obj.coords.r * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))

        # Left semicirlce
        psi = np.arctan(-cell_obj.coords.p_dx(cell_obj.coords.xl))

        th_l = np.linspace(-0.5 * np.pi + psi, 0.5 * np.pi + psi, num=numpoints_circle)
        cl_dx = cell_obj.coords.r * np.cos(th_l)
        cl_dy = cell_obj.coords.r * np.sin(th_l)

        cl_x = cell_obj.coords.xl - cl_dx
        cl_y = cell_obj.coords.p(cell_obj.coords.xl) + cl_dy

        # Split left semicircle into 2 halves at midpoint.
        frontcap_x = cl_x[:int(numpoints_circle / 2)]  # Get first half
        frontcap_y = cl_y[:int(numpoints_circle / 2)]

        endcap_x = cl_x[int(numpoints_circle / 2):]  # add final link to have a closed contour
        endcap_y = cl_y[int(numpoints_circle / 2):]

        # Right semicircle
        psi = np.arctan(-cell_obj.coords.p_dx(cell_obj.coords.xr))

        th_r = np.linspace(0.5 * np.pi - psi, -0.5 * np.pi - psi, num=numpoints_circle)
        cr_dx = cell_obj.coords.r * np.cos(th_r)
        cr_dy = cell_obj.coords.r * np.sin(th_r)

        cr_x = cr_dx + cell_obj.coords.xr
        cr_y = cr_dy + cell_obj.coords.p(cell_obj.coords.xr)

        x_all = np.concatenate((frontcap_x[::-1], x_top, cr_x[::-1], x_bot[::-1], endcap_x[::-1]))
        y_all = np.concatenate((frontcap_y[::-1], y_top, cr_y[::-1], y_bot[::-1], endcap_y[::-1]))

        return x_all, y_all

    from scipy.io import savemat
    from colicoords import Data, Cell, CellPlot
    import os, ast

    makedir(output_folder) #Make output directory

    cells = []
    cell_ids = []
    fit_errors = 0

    #Iterate through all images
    for i,result in enumerate(input_struct):
        masks = result['masks'] #Get masks from image
        filename = filenames[i] #Get corresponding filename to write
        savepath = os.path.join(output_folder,filename)

        #Itereate through all masks in image
        for j in range(masks.shape[2]):
            mask = masks[:,:,j]
            mask[mask >= 244] = 1  #Binarize

            # Calculate outline in cell coordinates and store
            data = Data()
            data.add_data(mask, 'binary')

            try:
                cell = Cell(data)
            except np.linalg.LinAlgError:
                fit_errors = fit_errors + 1

            cp = CellPlot(cell)

            xs, ys = calculate_outline(cp.cell_obj)

            left_x, left_y = xs[0:int(len(xs) / 2 + 1)], ys[0:int(len(ys) / 2 + 1)]
            right_x, right_y = xs[int(len(xs) / 2):], ys[int(len(ys) / 2):]

            right_x = np.append(right_x, left_x[0])  # Add overhangs
            right_y = np.append(right_y, left_y[0])

            assert left_x.shape == right_x.shape == left_y.shape == right_y.shape
            (i,) = left_x.shape

            mesh = np.zeros((4, i))
            mesh[0, :] = left_x
            mesh[1, :] = left_y
            mesh[2, :] = right_x[::-1]  # Rotate to match oufti format mesh orientation
            mesh[3, :] = right_y[::-1]

            mesh = np.transpose(mesh)  # Arrange to outfit format

            # Calculate box around the cell
            padding_x, padding_y = [10, 10]  # How many pixels of padding to add to box beyond cell extent
            midx, midy = np.mean(xs), np.mean(ys)  # Midpoint of cell
            cell_width, cell_height = xs.max() - xs.min(), ys.max() - ys.min()  # Cell extents

            x_corner = midx - cell_width / 2 - padding_x
            y_corner = midy - cell_height / 2 - padding_y
            box_width = cell_width + 2 * padding_x
            box_height = cell_height + 2 * padding_y

            if x_corner < 0:  # Resize boxes if partially outside image, to ensure the cell is always at centre
                box_width = box_width + x_corner
                x_corner = x_corner - x_corner
            if y_corner < 0:
                box_height = box_height + y_corner
                y_corner = y_corner - y_corner

            box = [x_corner, y_corner, box_width, box_height]

            cell_object = {'algorithm': 4, 'birthframe': 1, 'model': [], 'mesh': mesh, 'polarity': 0, 'stage': 1,
                           'timelapse': 0, 'divisions': [], 'box': box, 'ancestors': [], 'descendants': []}

            cells.append(cell_object)
            cell_ids.append(j)

        # Fill out struct fields with all information for this frame
        meshData = np.empty(1, dtype='object')
        meshData[0] = [cells]

        cellId = np.empty(1, dtype='object')
        cellId[0] = [cell_ids]

        cellList = {'cellId': cellId, 'meshData': meshData}

        cellListN = len(cell_ids)
        coefPCA = []
        mCell = []

        #Get parameters from supplied file
        p = {}
        with open(setfile) as fh:
            for line in fh:
                if line.startswith('%') or line.startswith('\n'):
                    continue  # Skip comment lines in file

                key, value = line.strip().split('=', 1)
                p[key] = value.strip()

                try:
                    p[key] = float(p[key])  # Convert read values to floats
                except ValueError:  # alternative handler for lists
                    p[key] = ast.literal_eval(p[key])

        rawPhaseFolder = []
        shiftfluo = [[0, 0], [0, 0]]
        shiftframes = []
        weights = []

        file = {'cellList': cellList, 'cellListN': cellListN, 'coefPCA': coefPCA, 'mCell': mCell, 'p': p,
                'rawPhaseFolder': rawPhaseFolder, 'shiftfluo': shiftfluo, 'shiftframes': shiftframes,
                'weights': weights}

        savemat(savepath, file)

















