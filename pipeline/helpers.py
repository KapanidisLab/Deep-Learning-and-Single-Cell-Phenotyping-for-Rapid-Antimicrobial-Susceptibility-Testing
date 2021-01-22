# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:17:56 2020

@author: Aleksander Zagajewski
"""

import matplotlib.pyplot as plt

from mrcnn import config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize as visualize

import numpy as np
import sklearn.metrics

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

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def class_id_to_name(dataset, ID):
    info = dataset.class_info
    for elm in info:
        id = elm['id']
        name = elm['name']
        if id == ID:
            return name
    raise ValueError('ID not found in dataset.')


def compute_pixel_metrics(dataset, image_ids, model):  # Custom function.

    print('Class ID and class name mappings...\n')
    print(dataset.class_info)
    print('')

    dataset.class_ids
    print('Pixelwise stats per image in test set...\n')

    gt_accumulator = np.asarray([], dtype='int')  # 1D arrays to store all pixels classifications throughout whole dataset
    pred_accumulator = np.asarray([], dtype='int')

    for image_id in image_ids:

        image = dataset.load_image(image_id)
        gt_mask, gt_class_id = dataset.load_mask(image_id)

        results = model.detect([image], verbose=0)


        (x, y, _) = gt_mask.shape  # extract shape
        gt_composite = np.zeros((x, y))
        pred_composite = np.zeros((x, y))

        for unique_class_id in np.unique(gt_class_id):
            idx = [i for i, cls in enumerate(gt_class_id) if cls == unique_class_id]  # Find matching indicies
            gt_masks_perclass = gt_mask[:,:,idx]  # extract masks per class
            pred_masks_perclass = results[0]['masks']

            assert ((gt_masks_perclass == 0) | (gt_masks_perclass == 1)).all()  # Assert masks are strictly binary
            assert ((pred_masks_perclass == 0) | (pred_masks_perclass == 1)).all()

            gt_sum = np.sum(gt_masks_perclass,
                            axis=2)  # Collapse instance masks into one mask of all instances of class
            pred_sum = np.sum(pred_masks_perclass, axis=2)

            gt_sum[gt_sum > 1] = 1  # Overlapping masks will produce summations >1. Threshold.
            pred_sum[pred_sum > 1] = 1

            gt_composite = gt_composite + gt_sum * unique_class_id  # Encode class into composite by CID
            pred_composite = pred_composite + pred_sum * unique_class_id

        gt_accumulator = np.append(gt_accumulator,gt_composite.flatten())  # Store across all images
        pred_accumulator = np.append(pred_accumulator,pred_composite.flatten())


    # Plot confusion matrix over all images
    label_names = [class_id_to_name(dataset, id) for id in np.unique(gt_accumulator)]
    cmat_total = sklearn.metrics.confusion_matrix(gt_accumulator.flatten(), pred_accumulator.flatten(), labels=np.unique(gt_accumulator), normalize='true')

    disp = sklearn.metrics.ConfusionMatrixDisplay(cmat_total, display_labels=label_names)
    disp.plot()
    plt.show()


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def compute_batch_ap(dataset, image_ids, config, model, verbose=1):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        APs.append(ap)
        if verbose:
            info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis, ...])
            print("{:3} {}   AP: {:.2f}".format(
                meta["image_id"][0], meta["original_image_shape"][0], ap))
    return APs


def image_stats(image_id, dataset):
    """Returns a dict of stats for one image."""
    image = dataset.load_image(image_id)
    mask, _ = dataset.load_mask(image_id)
    bbox = utils.extract_bboxes(mask)
    # Sanity check
    assert mask.shape[:2] == image.shape[:2]
    # Return stats dict
    return {
        "id": image_id,
        "shape": list(image.shape),
        "bbox": [[b[2] - b[0], b[3] - b[1]]
                 for b in bbox
                 # Uncomment to exclude nuclei with 1 pixel width
                 # or height (often on edges)
                 # if b[2] - b[0] > 1 and b[3] - b[1] > 1
                 ],
        "color": np.mean(image, axis=(0, 1)),
    }

