# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:17:56 2020

@author: Aleksander Zagajewski
"""

def dircounter(folder):
    '''
    Return number of directories downstream of folder, recursively.

    Parameters
    ----------
    folder : str
        Path to top.

    Returns
    -------
    counter : int
        Total dirs.

    '''

    import os
    if not os.path.exists(folder):
        raise ValueError('Folder does not exist.')

    counter = 0
    for root, dirs, files in os.walk(folder,topdown = True ):
            counter = counter + 1
    return counter 

def filecounter(folder):
    '''
    Return total number of files downstream of folder, recursively.

    Parameters
    ----------
    folder : str
        Path to top.

    Returns
    -------
    counter : int
        Total files.

    '''

    import os
    if not os.path.exists(folder):
        raise ValueError('Folder does not exist.')

    counter = 0
    for root, dirs, files in os.walk(folder,topdown = True ):
        for file in files:
            counter = counter + 1
    return counter 

def interspread(iterable, separator):
    
    '''
    Interspread iterable with separator between iterations.

    Parameters
    ----------
    iterable : array-like. Use strings.
        List of strings to be interspread
    separator : string
        Separator to interspread with.

    Returns
    ------
    output : array-like.
        string of iterable with separator interspread
    '''
    
    def interspread_gen(iterable, separator):
        it = iter(iterable)
        yield next(it)
        for x in it:
            yield separator
            yield x

    generator = interspread_gen(iterable, separator)

    output = ''
    while True:
        try:
            st = next(generator)
            output = output + st
        except StopIteration:
            return output


def makedir(path):  # Make directory if it doesn't exist yet.
    import os
    if not os.path.isdir(path):
        os.mkdir(path)


def get_parent_path(
        n):  # Generate correct parent directory, n levels up cwd. Useful for robust relative imports on different OS. 0 is the current cwd parent, 1 is the parent of the parent, etc
    import os
    assert n >= 0
    cwd = os.getcwd()
    parent = os.path.abspath(cwd)
    for order in range(0, n, 1):
        parent = os.path.dirname(parent)
    return (parent)


def im_2_uint16(image):  # Rescale and convert image to uint16.
    assert len(image.shape) == 2, 'Image must be 2D matrix '
    import numpy

    img = image.copy()  # Soft copy problems otherwise

    img = img - img.min()  # rescale bottom to 0
    img = img / img.max()  # rescale top to 1
    img = img * 65535  # rescale (0,1) -> (0,65535)
    img = numpy.around(img)  # Round
    img = img.astype('uint16')

    return img
