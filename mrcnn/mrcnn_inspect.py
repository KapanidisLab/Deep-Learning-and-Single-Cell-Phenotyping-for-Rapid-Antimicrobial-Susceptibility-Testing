# Import all the necessary libraries
import os
import datetime
import glob
import random
import sys
import re
import time
import warnings

import concurrent.futures

import matplotlib.pyplot as plt

import matplotlib.patches as patches
import skimage.io  # Used for imshow function
import skimage.transform  # Used for resize function
from skimage.morphology import label  # Used for Run-Length-Encoding RLE to create final submission

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

sys.stdout.flush()

from mrcnn import config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize as visualize

from mrcnn.visualize import display_images
from mrcnn.model import log

from mrcnn_train import BacDataset, BacConfig
from mrcnn_predict import PredictionConfig

import imgaug.augmenters as iaa  # import augmentation library
import sklearn.metrics

from pipeline.helpers import *

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


def compute_pixel_metrics(dataset, image_ids):  # Custom function.

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


def compute_batch_ap(dataset, image_ids, verbose=1):
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


def image_stats(image_id):
    """Returns a dict of stats for one image."""
    image = dataset_train.load_image(image_id)
    mask, _ = dataset_train.load_mask(image_id)
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



# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Log errors only.

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    WEIGHTSDIR = os.path.join(get_parent_path(0),'trial1_equalized_channels_dataset1lr=0.003_bs=2_t=20201205T1841', 'mask_rcnn_trial1_equalized_channels_dataset1lr=0.003_bs=2_t=.h5') # Foldername with weights. Prediction will run with latest weights.

    # create config
    config = PredictionConfig()  # Import test config
    config.IMAGES_PER_GPU = 1
    config.BATCH_SIZE = 1
    config.display()
    #config.DETECTION_NMS_THRESHOLD = 0.2
    #config.POST_NMS_ROIS_INFERENCE = 1000

    config_train = BacConfig()  # Import training config

    # TODO: Merge training and prediction configs for more robustness

    # Data for prediction
    testdir = os.path.join(get_parent_path(1), 'Data', 'Dataset1_05_12_20', 'Test')
    traindir = os.path.join(get_parent_path(1), 'Data', 'Dataset1_05_12_20', 'Train')


    dataset = BacDataset()
    dataset.load_dataset(testdir)
    dataset.prepare()

    dataset_train = BacDataset()
    dataset_train.load_dataset(traindir)
    dataset_train.prepare()

    print('----------------------------------------------------------')
    print('Train: ', len(dataset_train.image_ids))
    print('Test: ', len(dataset.image_ids))
    print('Weights: ' + WEIGHTSDIR)
    print('Class names: ', dataset.class_names)
    print('Class IDs', dataset.class_ids)
    print('----------------------------------------------------------')

    # ********************************************* INSPECT THE DATA ON TRAINING SET **************************************

    print('')
    print('----------------------------------------------------------')
    print('           INSPECTING THE DATA ON TRAINING SET')
    print('----------------------------------------------------------')

    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))

    # Display 1 random image
    image_ids = np.random.choice(dataset_train.image_ids, 1)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(skimage.img_as_ubyte(image), mask, class_ids, dataset_train.class_names, limit=1)

    # Collect image stats from the whole dataset

    # Loop through the dataset and compute stats over multiple threads
    # This might take a few minutes
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as e:
        stats = list(e.map(image_stats, dataset_train.image_ids))
    t_total = time.time() - t_start
    print("Total time: {:.1f} seconds".format(t_total))

    # Image stats

    image_shape = np.array([s['shape'] for s in stats])
    image_color = np.array([s['color'] for s in stats])
    print("Image Count: ", image_shape.shape[0])
    print("Height  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
        np.mean(image_shape[:, 0]), np.median(image_shape[:, 0]),
        np.min(image_shape[:, 0]), np.max(image_shape[:, 0])))
    print("Width   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
        np.mean(image_shape[:, 1]), np.median(image_shape[:, 1]),
        np.min(image_shape[:, 1]), np.max(image_shape[:, 1])))
    print("Color   mean (RGB): {:.2f} {:.2f} {:.2f}".format(*np.mean(image_color, axis=0)))

    # Histograms
    fig, ax = plt.subplots(3, 1)
    fig.suptitle('Image shape statistics', y=1)
    fig.tight_layout()

    ax[0].set_title("Height")
    ax[0].set_xlabel('Pixels')
    ax[0].set_ylabel('Freq Den')
    _ = ax[0].hist(image_shape[:, 0], bins=20)

    ax[1].set_title("Width")
    ax[1].set_xlabel('Pixels')
    ax[1].set_ylabel('Freq Den')
    _ = ax[1].hist(image_shape[:, 1], bins=20)

    ax[2].set_title("Height & Width")
    ax[2].set_xlabel('Pixels Width')
    ax[2].set_ylabel('Pixels Height')
    _ = ax[2].hist2d(image_shape[:, 1], image_shape[:, 0], bins=10, cmap="Blues")

    fig.subplots_adjust(top=0.65)

    plt.show()

    # Objects per image stats

    # Segment by image area
    image_area_bins = [256 ** 2, 600 ** 2, 1300 ** 2]

    print("Objects/Image")
    fig, ax = plt.subplots(len(image_area_bins), 1)
    fig.suptitle('Distribution of objects per image', y=1)
    fig.tight_layout()

    area_threshold = 0
    for i, image_area in enumerate(image_area_bins):
        objects_per_image = np.array([len(s['bbox'])
                                      for s in stats
                                      if area_threshold < (s['shape'][0] * s['shape'][1]) <= image_area])
        area_threshold = image_area
        if len(objects_per_image) == 0:
            print("Image area <= {:4}**2: None".format(np.sqrt(image_area)))
            continue
        print("Image area <= {:4.0f}**2:  mean: {:.1f}  median: {:.1f}  min: {:.1f}  max: {:.1f}".format(
            np.sqrt(image_area), objects_per_image.mean(), np.median(objects_per_image),
            objects_per_image.min(), objects_per_image.max()))
        ax[i].set_title("Image Area <= {:4}**2".format(np.sqrt(image_area)))
        ax[i].set_xlabel('Objects per image')
        ax[i].set_ylabel('Freq Den')
        _ = ax[i].hist(objects_per_image, bins=10)

    fig.subplots_adjust(top=0.65)
    plt.show()

    # Object size stats

    fig, ax = plt.subplots(1, len(image_area_bins))
    fig.suptitle('Object size statistics', y=1)
    fig.tight_layout()

    area_threshold = 0
    for i, image_area in enumerate(image_area_bins):
        object_shape = np.array([
            b
            for s in stats if area_threshold < (s['shape'][0] * s['shape'][1]) <= image_area
            for b in s['bbox']])

        try:

            object_area = object_shape[:, 0] * object_shape[:, 1]
            area_threshold = image_area

            print("\nImage Area <= {:.0f}**2".format(np.sqrt(image_area)))
            print("  Total Objects: ", object_shape.shape[0])
            print("  Object Height. mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
                np.mean(object_shape[:, 0]), np.median(object_shape[:, 0]),
                np.min(object_shape[:, 0]), np.max(object_shape[:, 0])))
            print("  Object Width.  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
                np.mean(object_shape[:, 1]), np.median(object_shape[:, 1]),
                np.min(object_shape[:, 1]), np.max(object_shape[:, 1])))
            print("  Object Area.   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
                np.mean(object_area), np.median(object_area),
                np.min(object_area), np.max(object_area)))

            # Show 2D histogram
            _ = ax[i].hist2d(object_shape[:, 1], object_shape[:, 0], bins=20, cmap="Blues")
            ax[i].set_xlabel('Object width (pix)')
            ax[i].set_ylabel('Object height (pix)')
            ax[i].set_title("Image Area <= {:4}**2".format(np.sqrt(image_area)))

        except IndexError as error:
            print("\nImage Area <= {:.0f}**2".format(np.sqrt(image_area)), '- No Objects Found!')

    fig.subplots_adjust(top=0.65)
    plt.show()

    # ******************************************* INSPECT AUGMENTATION ON TEST SET *****************************************

    print('')
    print('----------------------------------------------------------')
    print('           INSPECTING AUGMENTATION ON TRAIN SET')
    print('----------------------------------------------------------')

    # Define augmentation scheme
    warnings.filterwarnings('ignore', '', iaa.base.SuspiciousSingleImageShapeWarning, '',
                            0)  # Filter warnings from imgaug

    seq = [
        iaa.Fliplr(0.5),  # Flip LR with 50% probability
        iaa.Flipud(0.5),  # Flip UD 50% prob
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45))),  # Rotate up to 45 deg either way, 50% prob
        iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})),
        # Translate up to 20% on either axis independently, 50% prob
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),  # Gaussian convolve 50% prob
        #iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 65535))),  # up to 5% PSNR noise 50% prob
        iaa.Sometimes(0.5, iaa.Cutout(nb_iterations=(1, 10), size=0.05, squared=False, cval=0))
    ]

    augmentation = iaa.Sequential(seq)  # Execute in sequence from 1st to last

    limit = 4
    ax = get_ax(rows=2, cols=2)
    for i in range(limit):
        image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
            dataset_train, config_train, image_id, use_mini_mask=False, augment=False, augmentation=augmentation)
        image[image == 0] = 25000
        visualize.display_instances(skimage.img_as_ubyte(image), bbox, mask, class_ids,
                                    dataset.class_names, ax=ax[i // 2, i % 2],
                                    show_mask=False, show_bbox=False, title='Augmentation example {}'.format(i))

    plt.show()

    # ********************************************* INSPECT THE MODEL ON TEST SET *****************************************

    print('')
    print('----------------------------------------------------------')
    print('            INSPECTING THE MODEL ON TEST SET')
    print('----------------------------------------------------------')

    # define the model
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='inference', model_dir='./', config=config)
        model.load_weights(WEIGHTSDIR, by_name=True)

        image_id = 0 #Which image to inspect

        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        info = dataset.image_info[image_id]
        img_org_shape = modellib.parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0]

        # Run object detection

        results = model.detect(np.expand_dims(image, 0), verbose=0)

        # Display results
        r = results[0]
        print('')
        print('---GT---')
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

        print('')
        print('---RESULTS---')
        log('ROIs', r['rois'])
        log('Class_ID', r['class_ids'])
        log('Scores', r['scores'])
        log('Masks', r['masks'])

        # Compute AP over range 0.5 to 0.95 and print it
        utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                               r['rois'], r['class_ids'], r['scores'], r['masks'],
                               verbose=1)

        image_ubyte = skimage.img_as_ubyte(image)  # Convert to 8bit


        visualize.display_differences(
            image_ubyte,
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            dataset.class_names,
            show_box=False, show_mask=False,
            iou_threshold=0, score_threshold=0, ax=get_ax())
        plt.show()

        # Compute mAP
        print('')
        APs = compute_batch_ap(dataset, [image_id])
        print('')

        compute_pixel_metrics(dataset, [image_id]) #Compute pixel confusion mats

        # ************************************************* (1) RPN ********************************************************
        #
        # The Region Proposal Network (RPN) runs a lightweight binary classifier on a lot of boxes (anchors) over the image
        # and returns object/no-object scores. Anchors with high objectness score (positive anchors) are passed to the stage
        # two to be classified.
        #
        # Often, even positive anchors don't cover objects fully. So the RPN also regresses a refinement (a delta in
        # location and size) to be applied to the anchors to shift it and resize it a bit to the correct boundaries of
        # the object.

        # ============================================= (A) RPN targets ====================================================

        # The RPN targets are the training values for the RPN. To generate the targets, we start with a grid of anchors
        # that cover the full image at different scales, and then we compute the IoU of the anchors with ground truth
        # object. Positive anchors are those that have an IoU >= 0.7 with any ground truth object, and negative anchors
        # are those that don't cover any object by more than 0.3 IoU. Anchors in between (i.e. cover an object by IoU >=
        # 0.3 but < 0.7) are considered neutral and excluded from training.
        #
        # To train the RPN regressor, we also compute the shift and resizing needed to make the anchor cover the ground
        # truth object completely.

        print('*********************STEP BY STEP********************')
        print('')
        print('===================== RPN TRAIN =====================')
        print('')

        # Get anchors and convert to pixel coordinates
        anchors = model.get_anchors(image.shape)
        anchors = utils.denorm_boxes(anchors, image.shape[:2])
        log("anchors", anchors)

        # Generate RPN trainig targets
        # target_rpn_match is 1 for positive anchors, -1 for negative anchors
        # and 0 for neutral anchors.
        target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
            image.shape, anchors, gt_class_id, gt_bbox, model.config)
        log("target_rpn_match", target_rpn_match)
        log("target_rpn_bbox", target_rpn_bbox)

        positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
        negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
        neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
        positive_anchors = anchors[positive_anchor_ix]
        negative_anchors = anchors[negative_anchor_ix]
        neutral_anchors = anchors[neutral_anchor_ix]
        log("positive_anchors", positive_anchors)
        log("negative_anchors", negative_anchors)
        log("neutral anchors", neutral_anchors)

        # Apply refinement deltas to positive anchors
        refined_anchors = utils.apply_box_deltas(
            positive_anchors,
            target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
        log("refined_anchors", refined_anchors, )

        # Display positive anchors before refinement (dotted) and
        # after refinement (solid).
        visualize.draw_boxes(
            image_ubyte, ax=get_ax(),
            boxes=positive_anchors,
            refined_boxes=refined_anchors, title='RPN training - Positive anchors')
        plt.show()

        # Inspect negative anchors'
        visualize.draw_boxes(
            image_ubyte, ax=get_ax(),
            boxes=negative_anchors, title='RPN training - Negative anchors')
        plt.show()

        print('')
        print('==================== RPN PREDICTION ====================')
        print('')

        # ============================================= (B) RPN predictions ===============================================
        # Run RPN sub-graph
        pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

        # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
        if nms_node is None:
            nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
        if nms_node is None:  # TF 1.9-1.10
            nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

        rpn = model.run_graph(image[np.newaxis], [
            ("rpn_class", model.keras_model.get_layer("rpn_class").output),
            ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
            ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
            ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
            ("post_nms_anchor_ix", nms_node),
            ("proposals", model.keras_model.get_layer("ROI").output),
        ], image_metas=image_meta[np.newaxis])

        # Plot top 100 predictions at various stages
        limit = 100
        sorted_anchor_ids = np.argsort(rpn['rpn_class'][:, :, 1].flatten())[::-1]
        visualize.draw_boxes(image_ubyte, boxes=anchors[sorted_anchor_ids[:limit]], ax=get_ax(),
                             title='RPN predictions, top '
                                   '100 anchors. Before '
                                   'refinement. STEP 1.')
        plt.show()

        pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
        refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
        refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
        visualize.draw_boxes(image_ubyte, refined_boxes=refined_anchors_clipped[:limit], ax=get_ax(),
                             title='RPN predictions, '
                                   'top 100 anchors. '
                                   'After refinement '
                                   'and edge clipping. STEP 2')
        plt.show()

        ixs = rpn["post_nms_anchor_ix"][:limit]
        visualize.draw_boxes(image_ubyte, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax(),
                             title='RPN predictions, top '
                                   '100 anchors. After '
                                   'NMS. STEP 3')
        plt.show()

        # ************************************************* (2) CLASSIFIER *************************************************
        # Run a classfier on proposals

        print('')
        print('==================== CLASSIFIER ====================')
        print('')

        mrcnn = model.run_graph([image], [
            ("proposals", model.keras_model.get_layer("ROI").output),
            ("probs", model.keras_model.get_layer("mrcnn_class").output),
            ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
            ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ])

        # Proposals are in normalized coordinates
        proposals = mrcnn["proposals"][0]

        # Class ID, score, and mask per proposal
        roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
        roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
        roi_class_names = np.array(dataset.class_names)[roi_class_ids]
        roi_positive_ixs = np.where(roi_class_ids > 0)[0]

        # How many ROIs vs empty rows?
        print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
        print("{} Positive ROIs".format(len(roi_positive_ixs)))

        # Class counts
        print(list(zip(*np.unique(roi_class_names, return_counts=True))))

        # Display a random sample of proposals.
        # Proposals classified as background are dotted, and
        # the rest show their class and confidence score.
        limit = 200
        ixs = np.random.randint(0, proposals.shape[0], limit)
        captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                    for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
        visualize.draw_boxes(
            image_ubyte,
            boxes=utils.denorm_boxes(proposals[ixs], image.shape[:2]),
            visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
            captions=captions, title="ROIs Before Refinement. STEP 4.",
            ax=get_ax())

        plt.show()

        # Apply BBox refinement

        # Class-specific bounding box shifts.
        roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
        log("roi_bbox_specific", roi_bbox_specific)

        # Apply bounding box transformations
        # Shape: [N, (y1, x1, y2, x2)]
        refined_proposals = utils.apply_box_deltas(
            proposals, roi_bbox_specific * config.BBOX_STD_DEV)
        log("refined_proposals", refined_proposals)

        limit = 50 # Display 10 random positive proposals
        ids = np.random.randint(0, len(roi_positive_ixs), limit)

        captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                    for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
        visualize.draw_boxes(
            image_ubyte, ax=get_ax(),
            boxes=utils.denorm_boxes(proposals[roi_positive_ixs][ids], image.shape[:2]),
            refined_boxes=utils.denorm_boxes(refined_proposals[roi_positive_ixs][ids], image.shape[:2]),
            visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
            captions=captions, title="ROIs After Refinement. 50 random positives. STEP 5.")

        plt.show()

        # Filter low-confidence detections

        # Remove boxes classified as background
        keep = np.where(roi_class_ids > 0)[0]
        print("Keep {} detections:\n{}".format(keep.shape[0], keep))

        # Remove low confidence detections
        keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
        print("Remove boxes below {} confidence. Keep {}:\n{}".format(
            config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))

        # Per class non-max suppression

        # Apply per-class non-max suppression
        pre_nms_boxes = refined_proposals[keep]
        pre_nms_scores = roi_scores[keep]
        pre_nms_class_ids = roi_class_ids[keep]

        nms_keep = []
        for class_id in np.unique(pre_nms_class_ids):
            # Pick detections of this class
            ixs = np.where(pre_nms_class_ids == class_id)[0]
            # Apply NMS
            class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
                                                   pre_nms_scores[ixs],
                                                   config.DETECTION_NMS_THRESHOLD)
            # Map indicies
            class_keep = keep[ixs[class_keep]]
            nms_keep = np.union1d(nms_keep, class_keep)
            print("{:22}: {} -> {}".format(dataset.class_names[class_id][:20],
                                           keep[ixs], class_keep))

        keep = np.intersect1d(keep, nms_keep).astype(np.int32)
        print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))

        # Show final detections
        ixs = np.arange(len(keep))  # Display all
        # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
        captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                    for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
        visualize.draw_boxes(
            image_ubyte,
            boxes=utils.denorm_boxes(proposals[keep][ixs], image.shape[:2]),
            refined_boxes=utils.denorm_boxes(refined_proposals[keep][ixs], image.shape[:2]),
            visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
            captions=captions, title="Detections after NMS. STEP 6.",
            ax=get_ax())

        plt.show()

        print('ALL DONE!')
