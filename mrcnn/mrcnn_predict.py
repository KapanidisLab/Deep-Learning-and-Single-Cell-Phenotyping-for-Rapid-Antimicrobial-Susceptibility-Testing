# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:54:20 2020

@author: zagajewski
"""


# Import all the necessary libraries
import os
import datetime
import glob
import random
import sys
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function
from skimage.morphology import label                  #Used for Run-Length-Encoding RLE to create final submission
import skimage.measure
import scipy.spatial

import numpy as np
import pandas as pd

import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from skimage import img_as_bool
from skimage.transform import resize

from tqdm import tqdm


print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Skimage      :', skimage.__version__)
print('Scikit-learn :', sklearn.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
print('')

sys.stdout.flush()

from mrcnn import config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize


from mrcnn_train import BacDataset, BacConfig

from custom import Seg_to_OUFTI



class PredictionConfig(BacConfig): #Inherit from training config, perhaps make some changes
    NAME = 'bac_pred_cfg'
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 #Predict with batch size of 1

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    

# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    TOPDIR = r'.'
    WEIGHTSDIR = 'Early_augmentation_trials/trial_rotate+translate+defocus+noise+cutout_lr=0.003_bs=2_t=20200608T1653/'  # Foldername with weights. Prediction will run with latest weights.

    if type(WEIGHTSDIR) is str:  # If a folder is provided, find last weights and use that

        max = 0
        weights = os.path.join(TOPDIR, WEIGHTSDIR)
        valid_files = [files for files in os.listdir(weights) if files.endswith('.h5')]  # Pick files with h5 ext

        assert len(valid_files) == 1, 'More than one valid weight file found. Sorting not implemented yet.'
        weight_dir = os.path.join(TOPDIR, WEIGHTSDIR, valid_files[0])  # Path to latest weights

    # create config
    cfg = PredictionConfig()
    cfg.display()

    #Data for prediction and results dir
    testdir = os.path.join(TOPDIR,'Dataset_hafez_test')
    savedir = os.path.join(testdir, 'Results')

    figuredir = os.path.join(savedir, 'Segmentation maps')
    filedir = os.path.join(savedir, 'Files')

    for dir in [savedir,figuredir,filedir]:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print("Directory ", dir, " Created ")
        else:
            print("Directory ", dir, " already exists")

    test = BacDataset()
    test.load_dataset(testdir)
    test.prepare()

    print('----------------------------------------------------------')
    print('Test: ', len(test.image_ids))
    print('Weights: ' + weight_dir)
    print('Class names: ', test.class_names)
    print('Class IDs', test.class_ids)
    print('----------------------------------------------------------')


    # define the model
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='inference', model_dir='./', config=cfg)
        model.load_weights(weight_dir, by_name=True)

        for IID in tqdm(test.image_ids, desc='Segmenting:'): #Go through all images
            image = test.load_image(IID)
            results = model.detect(np.expand_dims(image, 0), verbose=0)

            r = results[0] #results comes out as a list in case of larger detect batch size
            masks = r['masks']


            #Save segmentation figures for view
            title = 'Segmentation map for image {}'.format(IID)
            colors = [(1, 0, 0, 1)] * len(r['class_ids']) #Plot detections in red 
            fig, ax = plt.subplots(1, 1, figsize=(16, 16))


            visualize.display_instances(skimage.img_as_ubyte(image), r['rois'], masks, r['class_ids'], test.class_names,
                              scores=r['scores'], title=title,
                              figsize=(16, 16), ax=ax,
                              show_mask=False, show_bbox=False,
                              colors=colors, captions=None)

            filename = os.path.join(figuredir,''.join([title,'.png']))
            Seg_to_OUFTI(masks)
            plt.savefig(filename,bbox_inches='tight')






    
    

