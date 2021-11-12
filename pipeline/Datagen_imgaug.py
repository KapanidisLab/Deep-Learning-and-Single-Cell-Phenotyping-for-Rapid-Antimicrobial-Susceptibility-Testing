# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:12:35 2021

@author: turnerp
"""

import numpy as np
from imgaug import augmenters as iaa
import cv2
import tensorflow as tf

seq1 = iaa.Sequential(
    [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        rotate=(-90, 90),
        mode="reflect"
        )
    ],
    random_order=True)

seq2 = iaa.Sequential(
    [
    iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5))),
    iaa.Sometimes(0.5, iaa.GammaContrast((0.5, 1.5))),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1))),
    ],
    random_order=True)


class DataGenerator(tf.keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, X_train, y_train,
                 batch_size=32, shuffle=True, augment=False):
        
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()


    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images = [self.X_train[k,:,:,:] for k in indexes]
        annots = [self.y_train[k,:] for k in indexes]

        X, y = self.__data_generation(images, annots)

        return X, y
    

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.X_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def augmentor(self, img):
        
        #rotation/translation augmentations, applied to images and masks
        seq_det1 = seq1.to_deterministic()
        image_aug = seq_det1.augment_images([img])[0]
    
        #changes in brightness and blurring, only applied to images
        seq_det2 = seq2.to_deterministic()
        image_aug = seq_det2.augment_images([image_aug])[0]

        return image_aug

    
    
    def __data_generation(self, images, annots, imshape):
        
        X = np.empty((self.batch_size, imshape[0], imshape[1], 1), dtype=np.float32)
        Y = np.empty((self.batch_size, label_size),  dtype=np.float32)

        for i, (img, annot) in enumerate(zip(images, annots)):

            if self.augment:
                img = self.augmentor(img)

            X[i,:,:,:] = img
            Y[i,:] = annot

        return X, Y
    
    
    
    
    
# create generator object

generator = DataGenerator(image_paths=train_data["Images"],
                    mask_paths=train_data["Masks"],
                    imshape = (1024,1024),
                    batch_size=10, augment=True)