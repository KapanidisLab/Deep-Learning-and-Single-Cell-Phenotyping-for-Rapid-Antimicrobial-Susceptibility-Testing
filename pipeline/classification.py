import copy

import keras
import numpy as np
import os
import sys

import skimage
import tensorflow
from scipy.ndimage.filters import gaussian_filter
from helpers import *
from Datagen_imgaug import DataGenerator
from imgaug import augmenters as iaa

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.optimizers import SGD, Adam, Nadam
import seaborn as sns
from matplotlib.colors import ListedColormap
import re, copy


def struct_from_file(dataset_folder=None, class_id = 1):

    #Loads cell segmentations from a dataset folder, creating a struct compatible with cells_from_struct.
    #Useful for creating additional classification data from manually annotated data used to train the 1st stage.
    #Dataset_folder must be properly formatted, this will not be checked.
    #Manual annotations have a confidence score of 1.0. Search paths propagated as absolute paths. Assigns class_id
    #as specified by user.

    import skimage.io
    from mrcnn.utils import extract_bboxes #Use the mask->bbox utility from first stage to make compatible bboxes

    output=[]

    #Search Train, Test, Val splits
    for split in os.listdir(dataset_folder):

        split = os.path.join(dataset_folder,split,'annots') #Add annots subfolder. Ignore raw images

        #Find all annotation folders in each split
        annots_dirs = [name for name in os.listdir(split) if os.path.isdir(os.path.join(split,name))]

        for annot_dir in annots_dirs:

            image_results = {'filename' : annot_dir} #Init storage object
            masks = []
            bboxes =[]

            annot_dir = os.path.join(split,annot_dir) #Propagate full path

            # Search each annotation folder for single-instance masks
            for file in os.listdir(annot_dir):

                assert file.endswith('.bmp'), 'Incorrect file detected. All single-instance masks must be .bmp'

                #Read mask
                readpath = os.path.join(annot_dir, file)

                try:
                    mask = skimage.io.imread(readpath)
                except ValueError:
                    print('WARNING - reader plugin error on {}, skipping mask.'.format(readpath))
                    continue

                mask[mask>=244] = 1 #Binarize mask and cast to bool
                assert mask.min() == 0 and mask.max() == 1

                masks.append(mask) #TODO Preallocate this for speed


            masks = np.asarray(masks, dtype='bool') #Convert to numpy arrays
            (N,x,y) = masks.shape

            masks = np.moveaxis(masks,0,-1) #Reshape masks from (N,x,y) -> (x,y,N) to match.

            bboxes = extract_bboxes(masks)  # Get bbox using original utility

            scores = np.ones(masks.shape[-1]) #Manual annotations have a confidence of 1.
            class_ids = np.ones(masks.shape[-1]) * class_id #Assign given class id

            image_results['masks'] = np.asarray(masks, dtype='bool') #Cast to bool for output to match
            image_results['rois'] = bboxes
            image_results['scores'] = np.asarray(scores, dtype='float32')
            image_results['class_ids'] = np.asarray(class_ids, dtype='int32')

            output.append(image_results) #Store image in output list

    return output


def apply_rois_to_image(input=None, mode=None, images=None):

    #Expects a results list as prepared by predict_mrcnn_segmenter. Applies rois from segmenter to supplied images and
    # returns single cell instances
    import copy
    output=[]

    for i,image_result in enumerate(input):
        image = images[i] #Get corresponding image
        image_cells = []

        if mode == 'masks':
            ROIs = image_result['masks']  # In mask mode, use masks directly to mask out image segments
            bboxes = image_result['rois']  # get bounding boxes to extract masked segments

        elif mode == 'bbox':  # In bbox mode, use bboxes instead to mask segments

            bboxes = image_result['rois']
            bbox_count = bboxes.shape()[0]

            ROIs = np.zeros((xlim, ylim, bbox_count))
            for i, box in enumerate(bboxes):
                ROI = np.zeros((xlim, ylim))
                [y1, x1, y2, x2] = box

                r = np.array([y1, y1, y2, y2])  # arrange vertices in clockwise order
                c = np.array([x1, x2, x2, x1])

                rr, cc = skimage.draw.polygon(r, c)  # Draw box
                ROI[rr, cc] = 1

                ROIs[:, :, i] = ROI  # Store as mask
        else:
            raise TypeError

        ROIs = ROIs.astype(int)  # Cast to int for matrix multiplication

        # Iterate through ROIs
        (x, y, N) = ROIs.shape
        for i in range(0, N, 1):

            [y1, x1, y2, x2] = bboxes[i]  # Get correct box
            masked_image = copy.deepcopy(image)  # Copy image to create mask

            ROI = ROIs[:, :, i]  # Fetch mask
            assert ROI.min() == 0 and ROI.max() == 1  # verify correct mask range

            ch_count = masked_image.shape[-1]  # Minimum of one trailing channel

            # Apply mask over all channels, elementwise multiplication
            for ch in range(0, ch_count, 1):
                masked_image[:, :, ch] = np.multiply(masked_image[:, :, ch], ROI)

            # Now extract the entire bbox of the masked image
            cell_instance = masked_image[y1:y2 + 1, x1:x2 + 1, :]  # Extract the bounding box of ROI

            # Add to output for this image
            image_cells.append(cell_instance)

        #Append to total output
        output.append(image_cells)

    return output


def cells_and_masks_from_struct(input=None, cond_IDs=None, image_dir=None, mode='masks'):

    # Expects a results list as prepared by predict_mrcnn_segmenter.
    # cond_IDs = list of strings with condition identifiers
    # image_dir = path to images prepared with Collect() and Sort()

    output = {'class_id_to_name': []}

    for i, cond_ID in enumerate(cond_IDs):  # Create output struct, populate with condition ids
        output[cond_ID] = []
        mapping = {'class_id': i,
                   'name': cond_ID}
        output['class_id_to_name'].append(mapping)

    for image_result in input:

        # Get condition ID from image result, try to match to supplied identifiers
        filename = image_result['filename']
        matched_condID = False
        matched = False

        for cond_ID in cond_IDs:

            pattern = cond_ID  # Assemble pattern
            pattern = re.escape(pattern)  # Auto escape any metacharacters inside cond_ID
            pattern = re.compile(pattern)  # Compile

            # If matched, get image from supplied image_dir
            if pattern.search(filename) is not None:
                if not matched_condID:

                    matched = True
                    matched_condID = cond_ID

                    image = fetch_image(image_dir, filename)  # fetch image() from helpers file
                    assert len(image.shape) == 2 or len(image.shape) == 3, 'Images must be either monochrome or RGB'

                    if len(image.shape) == 2:  # Add channel axis for monochrome images
                        image = np.expand_dims(image, -1)

                else:
                    raise TypeError('More than one cond_ID matched to image.')

        if matched is not True:
            raise RuntimeError('Image not matched to any supplied condition ID. Check input.')

        # Get instance masks. Use either segmentation masks of bounding boxes

        if mode == 'masks':
            ROIs = image_result['masks']  # In mask mode, use masks directly to mask out image segments
            bboxes = image_result['rois']  # get bounding boxes to extract masked segments
        elif mode == 'bbox':
            bboxes = image_result['rois']
            ROIs = np.zeros(image_result['masks'].shape)
            for i, bbox in enumerate(bboxes):
                [y1, x1, y2, x2] = bboxes[i]
                ROI = ROIs[:, :, i]
                ROI[y1:y2 + 1, x1:x2 + 1, :] = 1
                ROIs[:, :, i] = ROI

        else:
            raise ValueError('Mode not supported.')

        ROIs = ROIs.astype(int)  # Cast to int for matrix multiplication

        # Iterate through ROIs
        (x, y, N) = ROIs.shape
        for i in range(0, N, 1):

            [y1, x1, y2, x2] = bboxes[i]  # Get correct box
            masked_image = copy.deepcopy(image)  # Copy image to create mask

            ROI = ROIs[:, :, i]  # Fetch mask
            assert ROI.min() == 0 and ROI.max() == 1  # verify correct mask range

            ch_count = masked_image.shape[-1]  # Minimum of one trailing channel

            # Apply mask over all channels, elementwise multiplication
            for ch in range(0, ch_count, 1):
                masked_image[:, :, ch] = np.multiply(masked_image[:, :, ch], ROI)

            # Now extract the entire bbox of the masked image
            cell_instance = masked_image[y1:y2 + 1, x1:x2 + 1, :]  # Extract the bounding box of ROI

            #And mask
            mask_instance = ROI[y1:y2 + 1, x1:x2 + 1]
            # Add to output struct
            assert cell_instance.shape[0:2] == mask_instance.shape
            output[matched_condID].append([cell_instance,mask_instance])

    return output


def cells_from_struct(input=None, cond_IDs=None, image_dir=None, mode='masks'):
    #TODO reuse apply rois in second half of this

    #TODO This is the old version of the code, maintained here for backwards compatibility with scripts. Merge.

    #Expects a results list as prepared by predict_mrcnn_segmenter.
    # cond_IDs = list of strings with condition identifiers
    # image_dir = path to images prepared with Collect() and Sort()

    output = {'class_id_to_name' : [] }

    for i, cond_ID in enumerate(cond_IDs): #Create output struct, populate with condition ids
        output[cond_ID] = []
        mapping = {'class_id' : i,
                   'name' : cond_ID}
        output['class_id_to_name'].append(mapping)


    for image_result in input:

        #Get condition ID from image result, try to match to supplied identifiers
        filename = image_result['filename']
        matched_condID = False
        matched = False

        for cond_ID in cond_IDs:

            pattern = cond_ID #Assemble pattern
            pattern = re.escape(pattern) #Auto escape any metacharacters inside cond_ID
            pattern = re.compile(pattern) #Compile

            #If matched, get image from supplied image_dir
            if pattern.search(filename) is not None:
                if not matched_condID:

                    matched = True
                    matched_condID = cond_ID

                    image = fetch_image(image_dir, filename) #fetch image() from helpers file
                    assert len(image.shape) == 2 or len(image.shape) == 3, 'Images must be either monochrome or RGB'

                    if len(image.shape) == 2: #Add channel axis for monochrome images
                        image = np.expand_dims(image,-1)

                else:
                    raise TypeError('More than one cond_ID matched to image.')

        if matched is not True:
            raise RuntimeError('Image not matched to any supplied condition ID. Check input.')

        #Get instance masks. Use either segmentation masks of bounding boxes


        if mode =='masks':
            ROIs = image_result['masks']  #In mask mode, use masks directly to mask out image segments
            bboxes = image_result['rois'] #get bounding boxes to extract masked segments

        elif mode == 'bbox':
            bboxes = image_result['rois']
            ROIs = np.zeros(image_result['masks'].shape)
            for i, bbox in enumerate(bboxes):
                [y1, x1, y2, x2] = bboxes[i]
                ROI = ROIs[:, :, i]
                ROI[y1:y2 + 1, x1:x2 + 1, :] = 1
                ROIs[:, :, i] = ROI
        else:
            raise ValueError('Mode not supported.')

        ROIs = ROIs.astype(int)  #Cast to int for matrix multiplication

        #Iterate through ROIs
        (x,y,N) = ROIs.shape
        for i in range(0,N,1):

            [y1,x1,y2,x2] = bboxes[i] #Get correct box
            masked_image = copy.deepcopy(image) #Copy image to create mask

            ROI = ROIs[:,:,i] #Fetch mask
            assert ROI.min() == 0 and ROI.max() == 1 #verify correct mask range

            ch_count = masked_image.shape[-1] #Minimum of one trailing channel

            # Apply mask over all channels, elementwise multiplication
            for ch in range(0,ch_count,1):
                masked_image[:,:,ch] = np.multiply(masked_image[:,:,ch], ROI)

            #Now extract the entire bbox of the masked image
            cell_instance = masked_image[y1:y2+1, x1:x2+1, :] #Extract the bounding box of ROI

            #Add to output struct
            output[matched_condID].append(cell_instance)

    return output

def split_cell_sets(input=None, **kwargs):
    #Wrapper for sklearn train_test_split, stratifying split by total label distribution by default

    import sklearn.model_selection
    import collections

    total_cells = []
    total_ids = []

    for mapping in input['class_id_to_name']: #Concatanate across classes and store class index separately

        name = mapping['name'] #Get name
        cells = input[name] #Get corresponding cells
        ids = np.ones(len(cells)) * mapping['class_id'] #Write corresponding class id

        total_cells.extend(cells)
        total_ids.extend(ids)

    test_size = kwargs.get('test_size', False)
    if test_size == 0:
        X_train = total_cells
        y_train = total_ids

        X_test = []
        y_test = []
    elif test_size == 1.0:
        X_test = total_cells
        y_test = total_ids

        X_train = []
        y_train = []
    else:

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(total_cells, total_ids, stratify = total_ids, test_size = test_size, random_state = kwargs.get('random_state',False))

    #Print information

    counts_train = collections.Counter(y_train)
    counts_test = collections.Counter(y_test)

    for mapping in input['class_id_to_name']:
        name = mapping['name']
        id = mapping['class_id']

        print('Name: ' + name + ' mapped to ' + str(id))
        print('Train: ' + str(counts_train[id]))
        print('Test: ' + str(counts_test[id]))
        print('')



    return X_train, X_test, y_train, y_test

def save_cells_dataset(X_train=None, X_test=None, y_train=None, y_test=None, class_id_to_name=None, output_folder=None):

    import skimage.io

    def iterate(X,y,mode=None, pathmapping=None):

        if mode == 'Test':
            idx = 0
        elif mode =='Train':
            idx = 1
        else:
            raise TypeError

        #Iterate through all images
        for i,image in enumerate(X):
            class_id = y[i]

            #Match each image class id to a mapping. Get correct path and save.
            for ID in pathmapping:
                if class_id == ID:

                    path = pathmapping[ID][idx]
                    filename = pathmapping[ID][2] + str(i) + '.tif'
                    skimage.io.imsave(os.path.join(path,filename), image)



    #Saves cells dataset split between training and test sets

    #Create output dirs
    makedir(output_folder)
    test = os.path.join(output_folder,'Test')
    train = os.path.join(output_folder,'Train')

    makedir(test)
    makedir(train)

    category_ID_to_savepath = {}

    #Fill out a mapping object, linking class id and correct save path
    for mapping in class_id_to_name:
        cat_test = os.path.join(test,mapping['name'])
        cat_train = os.path.join(train,mapping['name'])

        makedir(cat_test)
        makedir(cat_train)

        category_ID_to_savepath[mapping['class_id']] = [cat_test, cat_train, mapping['name']] #Store savepaths and name


    #Operate on all both test and train sets
    iterate(X_train, y_train, mode='Train', pathmapping= category_ID_to_savepath)
    iterate(X_test, y_test, mode='Test', pathmapping=category_ID_to_savepath)


def define_model(mode = None, size_target=None, class_count=None, initial_lr=None, opt=None, init_source=None):
    #Defines and returns one of the included keras architectures. Init source either None for random init, or path to weights


    #Select weight source
    if init_source is None:
        weights = None
    else:
        weights = init_source


    #Select model from supported modes

    if mode == 'VGG16':
        model = VGG16(include_top=True, weights=weights, input_shape=size_target, classes = class_count)
    elif mode == 'ResNet50':
        model = ResNet50(include_top=True, weights=weights, input_shape=size_target, classes=class_count)
    elif mode == 'DenseNet121':
        model = DenseNet121(include_top=True, weights=weights, input_shape=size_target, classes=class_count)
    else:
        raise TypeError('Model {} not supported'.format(mode))

    for layer in model.layers:
        layer.trainable = True #Ensure all layers are trainable


    #Select optimimzer
    if opt == 'SGD+N': #SGD with nestrov
        optimizer = SGD(lr=initial_lr, momentum=0.9, nesterov=True) #SGD with nesterov momentum, no vanilla version
    elif opt == 'SGD': #SGD with ordinary momentum
        optimizer = SGD(lr=initial_lr, momentum=0.9, nesterov=False)  # SGD with nesterov momentum, no vanilla version
    elif opt == 'NAdam':
        optimizer = Nadam(lr=initial_lr)  # Nestrov Adam
    elif opt == 'Adam':
        optimizer = Adam(lr=initial_lr)  # Adam
    else:
        raise TypeError('Optimizer {} not supported'.format(opt))


    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train(mode = None, X_train = None, y_train = None, size_target = None, pad_cells=False, resize_cells=False, class_count = None, logdir = None, verbose=False, **kwargs):
    '''
    Trains vgg16 standard keras implementation on cells_object. Loads entire dataset into RAM by default for easier
    preprocessing (Change this is dataset size becomes too big). Random weight init.
    '''


    from keras.utils import to_categorical
    from skimage.transform import resize
    import keras.callbacks
    from datetime import datetime

    if pad_cells: assert not resize_cells
    elif resize_cells: assert not pad_cells

    #Get optional parameters, if not supplied load default values
    batch_size = kwargs.get('batch_size',16)
    epochs = kwargs.get('epochs',100)
    learning_rate = kwargs.get('learning_rate',0.001)
    dt_string = kwargs.get('dt_string', None)
    optimizer = kwargs.get('optimizer', 'SGD+N')
    init_source = kwargs.get('init_source',None)

    #Create model instance
    model = define_model(mode=mode, size_target=size_target, class_count=class_count, initial_lr=learning_rate, opt=optimizer, init_source=init_source)

    #Load and resize images, without maintaining aspect ratio.
    #One-hot encode labels

    n = [0,50,100,150,200,250]

    if verbose:
        inspect_model_data(X_train, y_train, n)


    if resize_cells:
        print('Resizing cell images to {}'.format(size_target))
        X_train = [resize(img, size_target) for img in X_train]
        X_train = [skimage.img_as_uint(img) for img in X_train]
    elif pad_cells:
        print('Padding cell images to {}'.format(size_target))
        X_train = [pad_to_size(img,size_target) for img in X_train]

    X_train = skimage.img_as_ubyte(np.asarray(X_train))
    y_train = to_categorical(y_train)

    if verbose:
        inspect_model_data(X_train, y_train, n)


    #Generator class. Compute pre-processing sttistics. Can also specify on the fly augmentation.
    validation_split = 0.2

    #Split apart validation
    count_total = len(X_train) #Total number of training+validation examples
    val_count = int((count_total * validation_split)) #Number of validation examples
    train_count = count_total - val_count #Number of actual training examples


    #Select indicies to include in validation
    val_idx = np.random.choice(np.arange(count_total), size=val_count, replace=False)
    train_idx = np.setdiff1d(np.arange(count_total), val_idx, assume_unique=False)

    val_X= X_train[val_idx,:,:,:]
    train_X = X_train[train_idx,:,:,:]

    val_y = y_train[val_idx,:]
    train_y = y_train[train_idx,:]

    assert len(val_X) + len(train_X) == count_total
    assert len(val_y) + len(train_y) == count_total


    #Rescale back to uint8 for better imgaug integration

    #Equalize all histograms
    histeq = iaa.Sequential([iaa.AllChannelsHistogramEqualization()])
    train_X = histeq(images = train_X)
    val_X = histeq(images = val_X)

    #Define augmentation

    seq1 = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                rotate=(-90, 90),
                mode="constant",
                cval=0
            )
        ],
        random_order=True)

    seq2 = iaa.Sequential(
        [
            iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5))), #Random sharpness increae
            iaa.Sometimes(0.5, iaa.WithChannels(0, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}))), #Random up to 10% misalignment
            iaa.Sometimes(0.5, iaa.MultiplyBrightness((0.5, 2.0))), #Brightness correction
            iaa.Sometimes(0.5, iaa.imgcorruptlike.GaussianNoise(severity=(1,2))), #Random gaussian noise
            #iaa.Sometimes(0.5, iaa.imgcorruptlike.DefocusBlur(severity=(1,2))), #Defocus correction
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.5)))
        ],
        )


    traingen = DataGenerator(train_X, train_y,
                 batch_size=batch_size, shuffle=True, augment=True, aug1=seq1, aug2=seq2)


    if verbose:
        inspect_model_data(np.asarray(traingen[0][0],dtype='uint8'), np.asarray(traingen[0][1],dtype='uint8'), [0,1])

    #Savefile name

    if dt_string is None:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M")

    checkpoint_name = dt_string + '.h5'

    #Create callbacks and learning rate scheduler. Reduce LR by factor 10 half way through

    def scheduler(epoch, lr):
        if epoch < round(epochs/2):
            return lr
        else:
            return learning_rate/10 #initial learning rate from outer scope divided by 10

    callbacks = [
        keras.callbacks.TensorBoard(log_dir=logdir,
                                    histogram_freq=0, write_graph=False, write_images=False),
        keras.callbacks.ModelCheckpoint(os.path.join(logdir,checkpoint_name),
                                        verbose=0, save_weights_only=False, save_best_only=True, monitor='loss',
                                        mode='min'),
        keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    ]

    #Train
    if not verbose:
        verbose_setting = 2
    else:
        verbose_setting = 1


    history = model.fit_generator(traingen, steps_per_epoch=len(traingen), validation_data=(val_X,val_y),  epochs=epochs, verbose=verbose_setting, callbacks=callbacks)

    #Plot basic stats
    summarize_diagnostics(history, checkpoint_name)

def simulate_defocus(img):
    '''
    Simulate microscope defocus using random Gaussian blurring
    '''

    #Draw for blurring event chance
    chance = np.random.uniform()

    if chance >= 0.6: #Blur in 40% of examples:

        #Draw for blurring magnitude
        sigma = np.random.uniform(0,2.5)

        return gaussian_filter(img,sigma=(sigma,sigma,0))
    else:
        return img

def rescale_intensity(img):
    p1, p99 = np.percentile(img, (1, 99))
    return skimage.exposure.rescale_intensity(img, in_range=(p1, p99))


def histeq16bit(image):
    '''16 bit histogram equalization'''

    def get_histogram(image, bins):
        try:
            """calculates and returns histogram"""
            # array with size of bins, set to zeros
            histogram = np.zeros(bins)
            # loop through pixels and sum up counts of pixels
            for pixel in image:
                histogram[pixel] += 1
        except IndexError as err:
            print('stop')
        return histogram

    def cumsum(a):
        """cumulative sum function"""
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)

        return np.array(b)

    """histogram equalisation for 16 bit images"""
    img = np.asarray(image)
    flat = img.flatten()
    hist = get_histogram(flat, (2 ** 16))
    cs = cumsum(hist)

    # numerator & denomenator
    nj = (cs - cs.min()) * (2 ** 16 - 1)
    N = cs.max() - cs.min()

    # re-normalize the cdf
    cs = nj / N
    cs = cs.astype('uint16')

    img_new = cs[flat]
    img_new = np.reshape(img_new, image.shape)

    return img_new




def pad_to_size(image,size_target):
    from skimage.transform import resize

    (sy,sx,ch) = image.shape

    if sy > size_target[0] or sx > size_target[1]:
        print('Resizing cell dimensions {} to {}'.format(image.shape, size_target))
        return skimage.img_as_uint(resize(image, size_target))

    total_sy = abs(sy-size_target[0])
    total_sx = abs(sx-size_target[1])

    if total_sy % 2 == 0:
        sy_before = total_sy/2
        sy_after = sy_before

    else:
        sy_before= int(total_sy/2)
        sy_after = sy_before + 1

    if total_sx % 2 == 0:
        sx_before = total_sx/2
        sx_after = sx_before

    else:
        sx_before= int(total_sx/2)
        sx_after = sx_before + 1

    new_img = np.zeros(size_target,dtype='uint16')

    for i in range(3):
        new_img[:,:,i] = np.pad(image[:,:,i], ((int(sy_before), int(sy_after)), (int(sx_before), int(sx_after))), mode='constant',constant_values=0)

    assert new_img.shape == size_target
    assert new_img.dtype == 'uint16'

    return new_img

def optimize(mode = None, X_train = None, y_train = None, parameter_grid = None, size_target = None, class_count = None, pad_cells=False, resize_cells=False, logdir = None):

    #Compute permutations of main parameters, call train() for each permutation. Wrap in multiprocessing to force GPU
    #memory release between runs, which otherwise doesn't happen

    import itertools, multiprocessing

    keysum = ['batch_size', 'learning_rate', 'epochs', 'optimizer']
    assert all([var in parameter_grid for var in keysum]), 'Check all parameters given'

    makedir(logdir) #Creat dir for logs

    for i, permutation in enumerate(itertools.product(parameter_grid['batch_size'], parameter_grid['learning_rate'], parameter_grid['epochs'], parameter_grid['optimizer'])):

        (batch_size, learning_rate, epochs, optimizer) = permutation #Fetch parameters
        dt_string = "{} BS {}, LR {}, epochs {}, opt {}".format(mode, batch_size, learning_rate, epochs, optimizer)

        #Create separate subdir for each run, for tensorboard ease

        logdir_run = os.path.join(logdir,dt_string)
        makedir(logdir_run)

        kwargs = {'mode': mode, 'X_train': X_train, 'y_train': y_train, 'batch_size': batch_size,
                  'learning_rate': learning_rate, 'epochs': epochs, 'size_target': size_target,
                  'class_count': class_count, 'logdir': logdir_run, 'optimizer': optimizer, 'dt_string': dt_string, 'verbose':False,
                  'pad_cells': pad_cells, 'resize_cells': resize_cells}

        p = multiprocessing.Process(target=train, kwargs=kwargs)
        p.start()
        p.join()

def inspect(modelpath=None, X_test=None, y_test=None, mean=None, size_target=None, class_id_to_name=None, pad_cells=False, resize_cells=False, normalise_CM=True, queue=None, colour_mapping=None):

    #Work on annotated (ie test) data.
    from keras.models import load_model
    from skimage.transform import resize
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    from skimage.transform import resize

    import matplotlib.pyplot as plt

    if pad_cells: assert not resize_cells
    elif resize_cells: assert not pad_cells

    result,_,model = predict(modelpath=modelpath,X_test=X_test, mean=mean, size_target=size_target, pad_cells=pad_cells, resize_cells=resize_cells)

    #Map classnames to class labels
    labels = [0]*len(class_id_to_name) #initialise array
    colour_mask = [0]*len(class_id_to_name)
    for elm in class_id_to_name:
        labels[elm['class_id']] = elm['name']
        colour_mask[elm['class_id']] = colour_mapping[elm['name']]





    #Plot matrix

    CM = confusion_matrix(y_test,result, normalize='true')
    CM_counts = confusion_matrix(y_test,result,normalize=None)



    #Seaborn plots sequential figures on top of each other. Use this to get multiple annotations

    CM_percentage = 100*CM
    processed_counts = CM_counts.flatten().tolist()
    processed_counts = ['({})'.format(elm) for elm in processed_counts]
    processed_counts = np.asarray(processed_counts).reshape((2,2))

    processed_percentage = np.asarray(np.rint(CM_percentage.flatten()),dtype='int').tolist()
    processed_percentage = ['{}%'.format(elm) for elm in processed_percentage]
    processed_percentage = np.asarray(processed_percentage).reshape((2,2))


    formatted_text = (np.asarray(["{}\n\n{}".format(
        data,text) for text, data in zip(processed_counts.flatten(), processed_percentage.flatten())])).reshape(2, 2)

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


    #display_misclassifications(result,y_test,X_test,class_id_to_name,10,resize_cells=resize_cells, pad_cells=pad_cells, size_target=size_target)

    if queue is not None:
        if normalise_CM == True:
            queue.put(CM)
        else:
            queue.put(CM_counts)

    return CM

def predict(modelpath=None, X_test=None, mean=None, size_target=None, pad_cells=False, resize_cells=False):
    import multiprocessing
    from keras import backend as K

    #Work on unannotated files

    from keras.models import load_model
    from skimage.transform import resize


    #Evaluate

    if pad_cells: assert not resize_cells
    elif resize_cells: assert not pad_cells

    # Load and pre-process data
    if resize_cells:
        X_test = [resize(img, size_target) for img in X_test]
    elif pad_cells:
        X_test = [pad_to_size(img,size_target) for img in X_test]

    # Load model
    if isinstance(modelpath,str):
        model = load_model(modelpath)
    elif isinstance(modelpath,keras.Model):
        model = modelpath
    else:
        raise TypeError('Modelpath is an unrecognised type.')

    X_test = skimage.img_as_ubyte(np.asarray(X_test))  # Cast between 0-1, resize

    histeq = iaa.Sequential([iaa.AllChannelsHistogramEqualization()]) #Equalize histogram
    X_test = histeq(images = X_test)

    # Subtract training mean
    X_test = X_test - mean

    result = model.predict(X_test)
    class_result = np.argmax(result,axis=1) #Decode from one-hot to integer
    class_confidence = np.max(result,axis=1)

    return class_result,class_confidence,model #Return result and model instance used


def display_misclassifications(result,y_test,X_test,class_id_to_name,n, resize_cells=None, pad_cells=None, size_target=None):

    from skimage.transform import resize

    # Find mismatches
    matching = np.asarray(y_test) == result
    idx = np.argwhere(matching == False).flatten()

    #Get ground truth and predicted classes
    mismatches_gt = np.asarray(y_test)[idx].flatten()
    mismatches_prediction = result[idx].flatten()


    if pad_cells:
        assert not resize_cells
    elif resize_cells:
        assert not pad_cells

    # Load and pre-process data
    if resize_cells:
        print('Resizing cell images to {}'.format(size_target))
        X_test = [resize(img, size_target) for img in X_test]
    elif pad_cells:
        print('Padding cell images to {}'.format(size_target))
        X_test = [pad_to_size(img, size_target) for img in X_test]


    #Fetch corresponding images, per class
    X_test = skimage.img_as_ubyte(np.asarray(X_test))  # Cast between 0-1, resize

    #Equalize all histograms
    histeq = iaa.Sequential([iaa.AllChannelsHistogramEqualization()])
    X_test= histeq(images = X_test)


    for i in range(len(class_id_to_name)):
        class_id = class_id_to_name[i]['class_id']
        name = class_id_to_name[i]['name']

        mismatches_perclass_idx = np.argwhere(mismatches_gt.flatten() == class_id).flatten()
        mismatches_perclass = X_test[idx[mismatches_perclass_idx]]
        mismatches_perclass_prediction = mismatches_prediction[mismatches_perclass_idx]

        # Create display handle
        rows = len(class_id_to_name) - 1
        cols = n

        fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(cols*2,rows*2))
        fig.suptitle('Ground truth {} missclassifications'.format(name), fontsize=12)

        active_col = 0
        for j in range(len(class_id_to_name)):
            class_id_mis = class_id_to_name[j]['class_id']
            if class_id_mis == class_id: continue


            name_mis = class_id_to_name[j]['name']

            matching_target = mismatches_perclass[np.argwhere(mismatches_perclass_prediction == class_id_mis)]

            try:
                random_n = matching_target[np.arange(n)]
                sample_count = len(random_n)
            except IndexError as err:
                random_n = matching_target #If fewer than 5 errors, just take what we have
                sample_count=len(random_n)

            for k in range(sample_count):
                img = random_n[k]
                img = img[0,:,:,:] #Remove singleton dimension

                #Calculate means - mask out zeros to get means within cell only
                means = get_masked_mean(img)
                means = np.asarray(means,dtype='int')

                img = skimage.img_as_ubyte(img)  # Recast for imshow

                title = str(name_mis)
                axs[active_col, k].imshow(img)
                axs[active_col, k].set_title(title)
                axs[active_col, k].set_xlabel(str(means))

            active_col += 1

        fig.set_constrained_layout_pads(hspace=0.1, wspace=0.1)
        plt.show()



def get_masked_mean(img):

    '''
    Get mean of image ignoring all zeros
    '''
    local_img = copy.deepcopy(img)

    mask = local_img == 0
    local_img_masked = np.ma.masked_array(local_img,mask=mask)
    mean = np.asarray(np.mean(local_img_masked,axis=(0,1)))

    return mean








