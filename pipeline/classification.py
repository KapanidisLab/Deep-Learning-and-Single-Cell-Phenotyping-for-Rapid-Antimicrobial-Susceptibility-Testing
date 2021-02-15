import numpy as np
import os
import sys


from helpers import *


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

            image_results = {'filename' : annot_dir+'.tif'} #Init storage object
            masks = []
            bboxes =[]

            annot_dir = os.path.join(split,annot_dir) #Propagate full path

            # Search each annotation folder for single-instance masks
            for file in os.listdir(annot_dir):

                assert file.endswith('.bmp'), 'Incorrect file detected. All single-instance masks must be .bmp'

                #Read mask
                readpath = os.path.join(annot_dir, file)

                mask = skimage.io.imread(readpath)
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


def cells_from_struct(input=None, cond_IDs=None, image_dir=None, mode='masks'):
    #TODO reuse apply rois in second half of this

    #Expects a results list as prepared by predict_mrcnn_segmenter.
    # cond_IDs = list of strings with condition identifiers
    # image_dir = path to images prepared with Collect() and Sort()

    import re, fnmatch, skimage.io, skimage.draw, copy

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

        elif mode =='bbox': #In bbox mode, use bboxes instead to mask segments

            bboxes = image_result['rois']
            bbox_count = bboxes.shape()[0]

            ROIs = np.zeros((xlim, ylim, bbox_count))
            for i,box in enumerate(bboxes):
                ROI = np.zeros((xlim,ylim))
                [y1,x1,y2,x2] = box

                r = np.array([y1,y1,y2,y2]) #arrange vertices in clockwise order
                c = np.array([x1,x2,x2,x1])

                rr, cc = skimage.draw.polygon(r, c) #Draw box
                ROI[rr,cc] = 1

                ROIs[:,:,i] = ROI #Store as mask
        else:
            raise TypeError

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

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(total_cells, total_ids, stratify = total_ids, test_size = kwargs.get('test_size', False), random_state = kwargs.get('random_state',False))

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


def define_model(mode = None, resize_target=None, class_count=None, initial_lr=None, opt=None, init_source=None):
    #Defines and returns one of the included keras architectures. Init source either None for random init, or path to weights


    from keras.applications.vgg16 import VGG16
    from keras.applications.resnet50 import ResNet50
    from keras.applications.densenet import DenseNet121
    from keras.optimizers import SGD, Adam, Nadam

    #Select weight source
    if init_source is None:
        weights = None
    else:
        weights = init_source


    #Select model from supported modes

    if mode == 'VGG16':
        model = VGG16(include_top=True, weights=weights, input_shape=resize_target, classes = class_count)
    elif mode == 'ResNet50':
        model = ResNet50(include_top=True, weights=weights, input_shape=resize_target, classes=class_count)
    elif mode == 'DenseNet121':
        model = DenseNet121(include_top=True, weights=weights, input_shape=resize_target, classes=class_count)
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


def train(mode = None, X_train = None, y_train = None, resize_target = None, class_count = None, logdir = None, **kwargs):
    '''
    Trains vgg16 standard keras implementation on cells_object. Loads entire dataset into RAM by default for easier
    preprocessing (Change this is dataset size becomes too big). Random weight init.
    '''


    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical
    from skimage.transform import resize
    import keras.callbacks
    from datetime import datetime

    #Get optional parameters, if not supplied load default values
    batch_size = kwargs.get('batch_size',16)
    epochs = kwargs.get('epochs',100)
    learning_rate = kwargs.get('learning_rate',0.001)
    dt_string = kwargs.get('dt_string', None)
    optimizer = kwargs.get('optimizer', 'SGD+N')

    #Create model instance
    model = define_model(mode=mode, resize_target=resize_target, class_count=class_count, initial_lr=learning_rate, opt=optimizer)

    #Load and resize images, without maintaining aspect ratio.
    #One-hot encode labels

    n = [0,50,100,150,200,250]

    inspect_model_data(X_train, y_train, n)

    X_train = [resize(img, resize_target) for img in X_train]
    X_train = np.asarray(X_train) #Cast between 0-1, resize
    y_train = to_categorical(y_train)

    inspect_model_data(X_train, y_train, n)


    #Generator class. Compute pre-processing sttistics. Can also specify on the fly augmentation.
    validation_split = 0.1

    datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, fill_mode='constant',
                                 cval=0, validation_split=validation_split, data_format='channels_last',
                                 horizontal_flip=True, vertical_flip=True, rotation_range=180,
                                 width_shift_range = 0.2, height_shift_range=0.2
                                 ) #10% validation split
    datagen.fit(X_train)

    #Create iterators
    train_it = datagen.flow(X_train, y=y_train, batch_size=batch_size, shuffle=True, seed=42, subset='training')
    val_it = datagen.flow(X_train, y=y_train, batch_size=batch_size, shuffle=True, seed=42, subset='validation')

    inspect_model_data(train_it.next()[0], train_it.next()[1], [0,1,2])

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
                                    histogram_freq=0, write_graph=False, write_images=True),
        keras.callbacks.ModelCheckpoint(os.path.join(logdir,checkpoint_name),
                                        verbose=0, save_weights_only=False, save_best_only=True, monitor='loss',
                                        mode='min'),
        keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    ]

    #Train
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=val_it, validation_steps=len(val_it), epochs=epochs, verbose=1, callbacks=callbacks)

    #Plot basic stats
    summarize_diagnostics(history, checkpoint_name)




def optimize(mode = None, X_train = None, y_train = None, parameter_grid = None, resize_target = None, class_count = None, logdir = None ):

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
                  'learning_rate': learning_rate, 'epochs': epochs, 'resize_target': resize_target,
                  'class_count': class_count, 'logdir': logdir_run, 'optimizer': optimizer, 'dt_string': dt_string}

        p = multiprocessing.Process(target=train, kwargs=kwargs)
        p.start()
        p.join()

def inspect(modelpath=None, X_test=None, y_test=None, mean=None, resize_target=None, class_id_to_name=None):

    #Work on annotated (ie test) data.

    from keras.models import load_model
    from skimage.transform import resize
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay

    import matplotlib.pyplot as plt

    result,model = predict(modelpath=modelpath,X_test=X_test, mean=mean, resize_target=resize_target)

    #Map classnames to class labels
    labels = [0]*len(class_id_to_name) #initialise array
    for elm in class_id_to_name:
        labels[elm['class_id']] = elm['name']

    #Plot matrix
    CM = confusion_matrix(y_test,result, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels = labels)
    disp.plot(cmap='Blues')
    plt.show()

def predict(modelpath=None, X_test=None, mean=None, resize_target=None):
    #Work on unannotated files

    from keras.models import load_model
    from skimage.transform import resize

    #Load model
    model = load_model(modelpath)

    #Load and pre-process data
    X_test = [resize(img, resize_target) for img in X_test]
    X_test = np.asarray(X_test)  # Cast between 0-1, resize

    #Subtract training mean
    X_test = X_test - mean

    #Evaluate
    result = model.predict(X_test)
    result = np.argmax(result,axis=1) #Decode from one-hot to integer

    return result,model #Return result and model instance used














