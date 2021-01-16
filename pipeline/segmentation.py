# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 19:33:56 2020

@author: Aleksander Zagajewski

This is an internal handle to integrate mrcnn functionality with the rest of the pipeline.
"""

from mrcnn import config  # import model
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

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

#-------------------------------------------MODEL CONFIGURATION CLASS---------------------------------------------------
class BacConfig(config.Config):
    #Inherit from base class, change settings where necessary

    NAME = 'BAC'

    LEARNING_RATE = 0.03
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1  # BG and Cell

    STEPS_PER_EPOCH = 100  # This is calculated later

    MEAN_PIXEL = np.array([0, 0, 0])

    IMAGE_CHANNEL_COUNT = 3

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # Anchor scales decreased to match size of bacteria better
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
#-----------------------------------------------------------------------------------------------------------------------


#----------------------------------------------MODEL DATASET CLASS------------------------------------------------------
class BacDataset(utils.Dataset):

    def load_dataset(self, dataset_dir, class_scheme):

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
            an_path = os.path.join(andir, os.path.splitext(filename)[
                0])  # Path to folder with image annotations, remove filename extension

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
    weights = kwargs.get('weights', False)
    output_folder = kwargs.get('output_folder', False)

    if not all([train_folder, validation_folder, config, weights, output_folder]):  # Verify input
        raise TypeError

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

    print('--------------------------------------------')
    print('Train: %d' % len(train_set.image_ids))
    print('Val: %d' % len(val_set.image_ids))
    print('Weights: ' + WEIGHTSDIR)
    print('--------------------------------------------')

    # Define augmentation scheme

    seq = [
        iaa.Fliplr(0.5),  # Flip LR with 50% probability
        iaa.Flipud(0.5),  # Flip UD 50% prob
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45))),  # Rotate up to 45 deg either way, 50% prob
        iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})),
        # Translate up to 20% on either axis independently, 50% prob
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),  # Gaussian convolve 50% prob
        # iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 65535))),  # up to 5% PSNR 50% prob
        iaa.Sometimes(0.5, iaa.Cutout(nb_iterations=(1, 10), size=0.05, squared=False, cval=0))
    ]

    augmentation = iaa.Sequential(seq)  # Execute in sequence from 1st to last

    # Define and train model
    print('Preparing to train...\n')

    warnings.filterwarnings('ignore', '', iaa.base.SuspiciousSingleImageShapeWarning, '',
                            0)  # Filter warnings from imgaug

    config.STEPS_PER_EPOCH = int(np.round(len(train_set.image_ids) / BS))  # Set 1 epoch = 1 whole pass
    config.VALIDATION_STEPS = int(np.round(len(val_set.image_ids) / BS)) # Iterate through entire validation set

    config.display()

    with tf.device(DEVICE):
        trmodel = modellib.MaskRCNN(mode='training', model_dir=output_folder, config=config)
        trmodel.load_weights(WEIGHTSDIR, by_name=True,
                             exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

        trmodel.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=50, layers='heads',
                      augmentation=augmentation)
        trmodel.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=100, layers='all',
                      augmentation=augmentation)
        trmodel.train(train_set, val_set, learning_rate=config.LEARNING_RATE / 10, epochs=150, layers='heads',
                      augmentation=augmentation)
        trmodel.train(train_set, val_set, learning_rate=config.LEARNING_RATE / 10, epochs=200, layers='all',
                      augmentation=augmentation)

