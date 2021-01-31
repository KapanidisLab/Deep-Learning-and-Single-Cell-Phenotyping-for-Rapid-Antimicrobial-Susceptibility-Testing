# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:17:28 2020

@author: Aleksander Zagajewski
"""

from helpers import *

def _multiproc_op(filename, operation, root, **kwargs): #Simple wrapper for image-wise operation to fit with Parallel
    import skimage.io, os
    img = skimage.io.imread(os.path.join(root,filename))
    img = operation(image = img, **kwargs)
    skimage.io.imsave(os.path.join(root,filename),img)
    return True


def TrainTestVal_split(**kwargs):
    '''
        Takes pre-prepared images and annotation masks and assembles a dataset folder. Supports multiple sources of
        images and annotations.

        Parameters
        ----------
        data_sources: list of strings
            list of paths to directory with image data. Data should be sorted and combined.
        annotation_sources: list of string
            path to directory with single cell masks, as generated by a generator.
        output_folder: string
            path to directory where output dataset structure will be created
        proportions: 3-tuple (x,y,z) of floats
            proportions into which split (train,val,test). x+y+z = 1
        seed: int
            seed for RNG.

        Returns
        -------

        '''

    data_sources = kwargs.get('data_sources', False)
    annotation_sources = kwargs.get('annotation_sources', False)
    output_folder = kwargs.get('output_folder', False)
    proportions = kwargs.get('proportions', False)
    seed = kwargs.get('seed', False)

    if not all([data_sources, annotation_sources, output_folder, proportions, seed]):  # Verify input
        raise TypeError

    assert len(data_sources) == len(annotation_sources), 'Each data source must have a corresponding annotation source'


    import os,numpy,fnmatch,random, distutils.file_util, distutils.dir_util
    from tqdm import tqdm

    (train_prop, test_prop, val_prop) = proportions
    assert train_prop+val_prop+test_prop == 1


    #Find total number of segmentation folders
    seg_folders = []

    for annotation_folder in annotation_sources:
        for folder in os.listdir(annotation_folder):
            sub = os.listdir(os.path.join(annotation_folder,folder))
            if sub == []:
                continue #Ignore empty folders with no masks
            seg_folders.append(os.path.join(annotation_folder,folder)) #Store both folder and path to it

    total = len(seg_folders)

    train_n, val_n, test_n = numpy.floor((train_prop *len(seg_folders), val_prop*len(seg_folders), test_prop*len(seg_folders)))
    train_n, val_n, test_n = int(train_n),int(val_n),int(test_n)

    matches = []

    #Now match annots and image paths into single structure
    for folder_path in seg_folders:
        for data_folder in data_sources:

            (_,folder) = os.path.split(folder_path)

            for image in os.listdir(data_folder):
                if fnmatch.fnmatch(image, folder+'.*'):

                    img = os.path.join(data_folder,image)
                    matches.append([folder_path,img])

    def to_tuple(lst):
        return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)
    matches = to_tuple(matches) #Recursevely convert to tuple so it can be hashed later


    #Randomly pick training set without replacement, then test set from rest
    random.seed(seed)

    Train = tuple(random.sample(matches,train_n))
    remaining = tuple(set(matches) - set(Train)) #Remaining set

    Test = tuple(random.sample(remaining,test_n))

    Validation = tuple(set(remaining) - set(Test)) #Assign rest to validation

    assert len(Train) + len(Test) +len(Validation) == total


    #Create output structure and copy files into it
    llist = {'Train': Train, 'Test': Test, 'Validation': Validation}

    makedir(output_folder)

    for key in llist:
        imagepath = os.path.join(output_folder, key,'images')
        annotpath = os.path.join(output_folder, key, 'annots')

        makedir(os.path.join(output_folder, key))
        makedir(imagepath)
        makedir(annotpath)

        imgset = llist[key]

        for elm in tqdm(imgset, total = len(imgset), unit = 'files', desc = 'Copying ' +str(key)+ ' files...'):
            tail = os.path.split(elm[0])[1]
            distutils.dir_util.copy_tree(elm[0],os.path.join(annotpath,tail))
            distutils.file_util.copy_file(elm[1],imagepath)

def Equalize_Channels(**kwargs):
    '''
    Subracts the mean of all tif images downstream of data_folder, and returns as float64 images.

    :param data_folder: string
        path from which to search down the tree
            cond_IDs: string
        condition_IDs to include

    :return:
    '''

    data_folder = kwargs.get('data_folder')
    cond_IDs = kwargs.get('cond_IDs')

    if not all([data_folder,cond_IDs]):
        raise TypeError

    import os, skimage.io, numpy

    intensity_total = numpy.zeros(3)
    pix_total = numpy.zeros(3)
    pix_min = numpy.zeros(3)
    pix_max = numpy.zeros(3)

    def _CalculateStats(data_folder): #Internal function to calculate stats

        intensity_total = numpy.zeros(3)
        pix_total = numpy.zeros(3)
        pix_min = numpy.zeros(3)
        pix_max = numpy.zeros(3)

        for folder in os.listdir(data_folder): #Calculate mean in first pass
            assert folder in cond_IDs, 'ERROR - A folder not matching given experimental condtions found in given directory. Aborting.'

            for file in os.listdir(os.path.join(data_folder,folder)):

                if not file.endswith('.tif') or file.endswith('.tiff'):
                    continue

                impath = os.path.join(data_folder,folder,file)

                image = skimage.io.imread(impath)

                (x,y,z) = image.shape

                pix_count = x*y #Total pixels in image

                for channel in range(z):
                    image_ch = image[:,:,channel]
                    ch_sum = image_ch.sum()
                    intensity_total[channel] = intensity_total[channel] + ch_sum #Sum total intensity per channel
                    pix_total[channel] = pix_total[channel] + x*y #Sum total pixels

                    if image_ch.min() < pix_min[channel]:
                        pix_min[channel] = image_ch.min()

                    if image_ch.max() > pix_max[channel]:
                        pix_max[channel] = image_ch.max()

            return intensity_total, pix_total, pix_min, pix_max

    intensity_total, pix_total, pix_min, pix_max = _CalculateStats(data_folder) #Calculate statistics across dataset

    #Now process all images
    average_pix = numpy.divide(intensity_total,pix_total) #Average pixel
    pix_min = pix_min - average_pix  # Min-Max values need updating too
    pix_max = pix_max - average_pix

    #Now process all images
    image_count = 0
    for folder in os.listdir(data_folder): #Subtract mean in second pass
        for file in os.listdir(os.path.join(data_folder,folder)):
            if not file.endswith('.tif') or file.endswith('.tiff'):
                continue

            impath = os.path.join(data_folder, folder, file)
            image = skimage.io.imread(impath) #Read image


            #Rescale to (a,b) = (-1,1)
            a = -1
            b = 1

            with numpy.errstate(divide='ignore'):
                #image = a + numpy.divide(((image-pix_min)*(b-a)), (pix_max - pix_min))  # Rescale to [-1,1]
                image = numpy.divide((image - average_pix), (pix_max-pix_min))
                image[~ numpy.isfinite(image)] = 0  # -inf inf NaN



            skimage.io.imsave(impath,image,check_contrast=False)
            image_count = image_count + 1

    print('Average pixel', str(average_pix), 'subtracted from', str(image_count), 'images')

    #Verify that data has been processed correctly

    intensity_total, pix_total, pix_min, pix_max = _CalculateStats(data_folder)  # Calculate statistics across dataset
    average_pix = intensity_total/pix_total
    tolerance = numpy.asarray([1e-4, 1e-4, 1e-4])**2

    assert all(average_pix<=tolerance) #Assert average is 0




def SortNIM(data_folder, output_folder = None, **kwargs):
    
    '''
    Sorts through NIM default file structure and sorts files into subfolders based on condtion type.
    
    Parameters
    ----------
    path : string
        path to directory with image data
    dims : Array-like of ints of length 3
        img_dimes = (sx,sy,sz), the dimensions of each image to sort
    cond_IDs : Array-like of strings
        cond_IDs = ['WT', 'RIF', ... ], where entries correspond to condition IDs in filenames.
    image_channels  : Array like of strings
        image_channels = ['BF', 'DAPI', ...], where entries correspond to image channels

    Returns
    -------
    output_folder: string
        absolute path to folder where data was written
    '''
    
    img_dims = kwargs.get('dims', False)
    cond_IDs = kwargs.get('cond_IDs', False)
    image_channels = kwargs.get('image_channels', False)
    
    if not all([data_folder,img_dims,cond_IDs]): #Verify input
        raise TypeError

    assert len(img_dims) == 3, 'img_dims must have exactly 3 values'
    assert cond_IDs

    image_channels = list(set(image_channels)) #Extract unique channels


    
    import os,skimage.io, warnings,numpy

    if output_folder is None: #Default output path if none provided
        output_folder = os.path.join(data_folder,'Segregated')
    else:
        assert type(output_folder) == str

    
    (sx,sy,sz) = img_dims

    NIM = True
     
    from tqdm import tqdm
    import re
        
    
    makedir(output_folder)
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    for root, dirs, files in tqdm(os.walk(data_folder, topdown=True), total = dircounter(data_folder), unit = 'dirs', desc = 'Searching directories'):
        for file in files:
    
            if file.endswith("posZ0.tif"): #Find all image files in folder
                file_delim = file.split('_') #Split filename to find metadata
                
    
                img = skimage.io.imread(os.path.join(root,file)) #Load image
                
                assert img.shape == (sz, sx, sy), 'Images not of usual dimension'
                
                img = numpy.mean(img, axis = 0) #Average frames
                assert img.shape == (sx,sy), 'Images averaged wrongly'
                
                if NIM == True:
                    img = img[:,0:int(sy/int(2))] #Crop FoV to remove unused half
                    assert img.shape ==(sx,int(sy/2)), 'Images cropped incorrectly'
               
                img = im_2_uint16(img) #Convert to uint16
                
                for condition_ID in cond_IDs: #Iterate over expected conditions and save files
                    
                    if file_delim[1] == condition_ID:
                        savefolder = os.path.join(output_folder,condition_ID) #Append cond_ID
                        makedir(savefolder)
    
                        imtype = file_delim[3] #Get image type identifier

                        for channel in image_channels: #Iterate over all expected image channels

                            pattern = re.compile(channel)
                            if re.search(pattern, imtype) is not None:
                                savefolder = os.path.join(savefolder, channel) #Create save directory for each matched channel
                                makedir(savefolder)

                                                                            
                                filename = [file_delim[part]+'_' for part in range(0,7,1)]  #Assemble filename
                                filename = ''.join(filename) + '.tiff'
                        
                                skimage.io.imsave(os.path.join(savefolder,filename),img) #Write image to appropriate sub folder
                        
    return output_folder
                        

def CollectNIM(data_folder, output_folder = None, **kwargs):
    
    cond_IDs = kwargs.get('cond_IDs', False)
    image_channels = kwargs.get('image_channels', False)
    
    if not all([data_folder,cond_IDs, image_channels]): #Verify input
        raise TypeError

    import os, numpy, skimage.io, sys
    from tqdm import tqdm

    if output_folder is None:
        output_folder = os.path.join(data_folder,'Combined')
    else:
        assert type(output_folder) == str


    makedir(output_folder)

    success_counter_store = []
    total_store = []
    
    for cond_ID in cond_IDs:
        folder = os.path.join(data_folder,cond_ID)
        
        output_folder_cond = os.path.join(output_folder,cond_ID)
        makedir(output_folder_cond)

        channel_paths = [os.path.join(folder,channel) for channel in image_channels]


        success_counter = 0
        total = 0


        for root, dirs, files in os.walk(channel_paths[0]): #use first channel to build comparison scaffold
            for file in tqdm(files, desc= cond_ID, total = filecounter(channel_paths[0])):
                
                
                file_delim = file.split('_')
                prefix = file_delim[0]
                condition = file_delim[1]
                channel = file_delim[3]
                FOV = file_delim[4]

                if condition != cond_ID is None:
                    continue #Continue to next file if conditions do not match


                dataset_tag = [int(s) for s in list(channel) if s.isdigit()] #Extract dataset tag from channel info
                assert len(dataset_tag) == 1 #There should only be one numeric tag per folder

                image = skimage.io.imread(os.path.join(root, file)) #Load image to match
                (sx, sy) = image.shape #
                combined_image = numpy.zeros((3,sx,sy),dtype = 'uint16') #Initialize combined images of (c,x,y) format, where c is the number of channels to match
                combined_image[0,:,:] = im_2_uint16(image) #Populate first channel

                matches = 0
                total = total + 1

                for i in range(1,len(channel_paths),1): #Iterate over remaining channels to match the image
                    for root2, dirs2, files2 in os.walk(channel_paths[i]): #examine other channels


                        for file2 in files2:

                            file2_delim = file2.split('_')
                            condition2 = file2_delim[1]
                            channel2 = file2_delim[3]
                            FOV2 = file2_delim[4]

                            dataset_tag2 = [int(s) for s in list(channel2) if s.isdigit()] #Extract dataset tag from channel info
                            assert len(dataset_tag2) == 1  # There should only be one numeric tag per folder

                            if condition == condition2 and FOV == FOV2 and dataset_tag == dataset_tag2:
                                image2 = skimage.io.imread(os.path.join(root2, file2))
                                assert image2.shape == image.shape #All images must be the same size

                                combined_image[i,:,:] = im_2_uint16(image2) #Populate remaining channels
                                if len(channel_paths) != 1:

                                    matches = matches + 1 #Keep track of matched channels

                if matches == len(channel_paths)-1: #Fully matched images only. Matches are one less than total channel number

                    filename = prefix + '_combined_' + str(dataset_tag[0]) + '_' +str(cond_ID)+'_' + FOV + '.tif'  # Assemble filename
                    savepath = os.path.join(output_folder_cond, filename)

                    skimage.io.imsave(savepath, combined_image, check_contrast=False)  # Write to file

                    success_counter = success_counter + 1


            success_counter_store.append(success_counter)
            total_store.append(total)

    #At the end, print a summary 
            
    sys.stdout.flush()
    print('')
    for count,elem in enumerate(success_counter_store):
        print(cond_IDs[count], ' : ' ,elem, '/', total_store[count], ' matched successfully' )
    sys.stdout.flush()
    
    
    return output_folder


def BatchProcessor(data_folder,operation, op, **kwargs):
    '''
    Utility function for multithreading operations. 
    Iterates through all files in data_folder, applies operation to each
    file, passes kwargs to operation.
    
    Parallel processing enabled via joblib and default loky backend.

    Parameters
    ----------
    data_folder : str
        Path to images. All files downstream are read.
    operation : function
        image-wise operation to perform on the images in data_folder
    op : str
        key reference to op in factory, for progress bar display only
    **kwargs : **kwargs
        Arguments other than image to pass to operation

    Returns
    -------
    None.

    '''
    import os
    from tqdm import tqdm
    
    import multiprocessing
    from joblib import Parallel, delayed
    
    num_cores = multiprocessing.cpu_count()
    
    for root, dirs, files in os.walk(data_folder, topdown=True):
        
        directory = os.path.basename(root)
        if directory == os.path.basename(data_folder):
            continue #Skip first iteration of recursive os.walk which only picks up sub-dirs
        
       
        
        #Provide a progress bar
        inputs = tqdm(files, total = filecounter(root), unit = 'files', desc = op + ' : ' + directory) 
        
        #Heavy-lifting
        output = Parallel(n_jobs=num_cores)(delayed(_multiproc_op)(file,operation,root,**kwargs) for file in inputs)


def Imadjust(**kwargs):
    
    image = kwargs.get('image', False )
    if image is False:
        raise TypeError
        
    if len(image.shape) == 3: #If given a stack, work on only the first (assumed to be BF)
        index = kwargs.get('index', False)
        if index is False:
            raise TypeError
        image = image[index,:,:] 
    
    import numpy
    from skimage import exposure
    
    img = image.copy()
    
    v_min, v_max = numpy.percentile(img, (1, 99))
    img = exposure.rescale_intensity(img, in_range=(v_min, v_max))
    return img
    
def Iminvert(**kwargs):
    image = kwargs.get('image', False )
    if image is False:
        raise TypeError
        
    if len(image.shape) == 3: #If given a stack, work on only the first (assumed to be BF)
        index = kwargs.get('index', False)
        if index is False:
            raise TypeError
        image = image[index,:,:] 
        
    from skimage import util 
    inverted_img = util.invert(image)
    return inverted_img
