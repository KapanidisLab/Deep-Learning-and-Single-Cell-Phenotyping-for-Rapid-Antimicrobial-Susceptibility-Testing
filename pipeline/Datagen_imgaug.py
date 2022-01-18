# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:12:35 2021

@author: turnerp
"""

import numpy as np
import keras



class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, X_train, y_train,
                 batch_size=32, shuffle=True, augment=False, aug1 = None, aug2 = None):

        print('DataGenerator - supplied {} images, batch size {}'.format(len(X_train), batch_size))
        print('DataGenerator - approx {} batches available.'.format(np.ceil(len(X_train)/batch_size)))

        self.X_train = X_train
        self.y_train = y_train

        (N,ypix,xpix,ch) = X_train.shape
        self.imshape = (ypix,xpix,ch)

        print('DataGenerator - generating images of size {}'.format(self.imshape))

        if len(y_train.shape) == 1:
            self.label_size = 1
        else:
            (_,self.label_size) = y_train.shape

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.aug1 = aug1
        self.aug2 = aug2

        self.on_epoch_end()



    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.X_train) / self.batch_size))


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images = [self.X_train[k,:,:,:] for k in indexes]

        if self.label_size == 1:
            annots = [self.y_train[k] for k in indexes]
        else:
            annots = [self.y_train[k,:] for k in indexes]

        X, y = self.__data_generation(images, annots, self.imshape)

        return X, y
    

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.X_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def augmentor(self, img):
        
        #rotation/translation augmentations, applied to images and masks
        if self.aug1 is not None:
            seq_det1 = self.aug1.to_deterministic()
            image_aug = seq_det1.augment_images([img])[0]
    
        #changes in brightness and blurring, only applied to images

        if self.aug2 is not None:
            seq_det2 = self.aug2.to_deterministic()
            image_aug = seq_det2.augment_images([image_aug])[0]

        return image_aug

    
    
    def __data_generation(self, images, annots, imshape):
        
        X = np.empty((self.batch_size, imshape[0], imshape[1], imshape[2]), dtype=np.float32)
        Y = np.empty((self.batch_size, self.label_size),  dtype=np.float32)

        for i, (img, annot) in enumerate(zip(images, annots)):

            if self.augment:
                img = self.augmentor(img)

            X[i,:,:,:] = img
            Y[i,:] = annot

        return X, Y
    
    
    
    
    
