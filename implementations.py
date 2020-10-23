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


def SortNIM(data_folder,**kwargs):
    
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
    
    
    import os,skimage.io, warnings,numpy

    output_folder = os.path.join(data_folder,'Segregated')
    
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
                        

def CollectNIM(data_folder,**kwargs):
    
    cond_IDs = kwargs.get('cond_IDs', False)
    image_channels = kwargs.get('image_channels', False)
    
    if not all([data_folder,cond_IDs, image_channels]): #Verify input
        raise TypeError
        
    import os, numpy, skimage.io, sys, re
    from tqdm import tqdm
    
    output_folder = os.path.join(data_folder,'Combined')
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
    print('\n')
    print('')
    print('-------------------------------------------')
    for count,elem in enumerate(success_counter_store):
        print(cond_IDs[count], ' : ' ,elem, '/', total_store[count], ' matched successfully' )
    print('-------------------------------------------')
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
    import skimage.io, os
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
