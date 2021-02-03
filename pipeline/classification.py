import numpy as np
import os
import sys


from helpers import *


def struct_from_file(dataset_folder=None, class_id = 0):

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


            masks = np.asarray(masks) #Convert to numpy arrays
            (N,x,y) = masks.shape
            masks = np.reshape(masks,(x,y,N)) #Reshape masks from (N,x,y) -> (x,y,N) to match.

            bboxes = extract_bboxes(masks)  # Get bbox using original utility

            scores = np.ones(masks.shape[-1]) #Manual annotations have a confidence of 1.
            class_ids = np.ones(masks.shape[-1]) * class_id #Assign given class id

            image_results['masks'] = np.asarray(masks, dtype='bool') #Cast to bool for output to match
            image_results['rois'] = bboxes
            image_results['scores'] = np.asarray(scores, dtype='float32')
            image_results['class_ids'] = np.asarray(class_ids, dtype='int32')

            output.append(image_results) #Store image in output list

    return output


def cells_from_struct(input=None, cond_IDs=None, image_dir=None, mode='masks'):

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


def train_vgg16(X_train, y_train, resize_target = None, class_count = None, logdir = None):
    '''
    Trains vgg16 standard keras implementation on cells_object. Loads entire dataset into RAM by default for easier
    preprocessing (Change this is dataset size becomes too big). Random weight init.
    '''


    from keras.applications.vgg16 import VGG16
    from keras.optimizers import Adam, SGD
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical
    from skimage.transform import resize
    import keras.callbacks
    from datetime import datetime

    def define_model(resize_target=None, class_count=None):

        model = VGG16(include_top=True, weights=None, input_shape=resize_target, classes = class_count)

        for layer in model.layers:
            layer.trainable = True #Ensure all layers are trainable

        #optimizer = Adam(lr=0.01) #Adam with Keras default hyperparameters
        optimizer = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    #Load and resize images, without maintaining aspect ratio.
    #One-hot encode labels

    X_train = [resize(img, resize_target) for img in X_train]
    X_train = np.asarray(X_train)

    y_train = to_categorical(y_train)

    #Create model instance
    model = define_model(resize_target=resize_target, class_count=class_count)

    #Generator class. Compute pre-processing sttistics. Can also specify on the fly augmentation.
    validation_split = 0.1
    batch_size = 16

    datagen = ImageDataGenerator(featurewise_center=False, fill_mode='nearest', validation_split=validation_split, data_format='channels_last') #10% validation split
    datagen.fit(X_train)

    #Create iterators
    train_it = datagen.flow(X_train, y=y_train, batch_size=batch_size, shuffle=True, seed=42, subset='training')
    val_it = datagen.flow(X_train, y=y_train, batch_size=batch_size, shuffle=True, seed=42, subset='validation')


    #Create callbacks

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M")
    checkpoint_name = 'VGG16_'+dt_string+'.h5'

    callbacks = [
        keras.callbacks.TensorBoard(log_dir=logdir,
                                    histogram_freq=0, write_graph=True, write_images=True),
        # EDIT - set write images to True
        keras.callbacks.ModelCheckpoint(os.path.join(logdir,checkpoint_name),
                                        verbose=0, save_weights_only=True, save_best_only=True, monitor='loss',
                                        mode='min'),
    ]

    #Train
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=val_it, validation_steps=len(val_it), epochs=50, verbose=1, callbacks=callbacks)

    #Plot basic stats
    summarize_diagnostics(history)


    #Create iterators from file
    #train_it = datagen.flow_from_directory(traindir,subset='training', class_mode='categorical', batch_size=16, target_size=(224, 224), color_mode="rgb", shuffle=True, seed =42)
    #val_it = datagen.flow_from_directory(traindir,subset='validation', class_mode='categorical', batch_size=16, target_size=(224, 224), color_mode="rgb", shuffle=True, seed =42)
