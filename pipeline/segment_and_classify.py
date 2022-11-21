
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.measure import find_contours
from segmentation import *
from classification import *
from helpers import *
import os

from segment_classify_distribution import segment_and_classify

def plot_detections(segmentations=None,classifications=None,images=None, mappings=None, show_caption=True, title='', save=None, savepath=None):

    assert len(segmentations) == len(classifications)

    for i,seg in enumerate(segmentations):

        fig, ax = plt.subplots(1,2, figsize=(16,16), constrained_layout=True)
        fig2, ax2 = plt.subplots(1,1,figsize=(16,16), constrained_layout=True)

        boxes = seg['rois']
        scores = seg['scores']
        masks = seg['masks']
        phenotypes = classifications[i]
        image = images[i]

        N = boxes.shape[0]

        for j in range(N):
            y1, x1, y2, x2 = boxes[j]
            score = scores[j]
            mask = masks[:,:,j]
            phenotype = phenotypes[j]


            colour = mappings[phenotype]['colour']
            name = mappings[phenotype]['name']

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=colour,linewidth=1.5)
                ax[0].add_patch(p)

                #Copy over to other figure

                pp = Polygon(verts, facecolor="none", edgecolor=colour,linewidth=1.5)
                ax2.add_patch(pp)

        #Caption
        if show_caption:
            caption = "{}".format(name)
            ax[0].text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="none")

        #Plot original image unchanged, and beside it grayscaled and with annotations
        image = img_as_ubyte(image) #8bit conversion
        ax[1].imshow(image)

        ax[0].imshow(rgb2gray(image),cmap=plt.cm.gray)

        if title:
            fig.suptitle(title, fontsize=40)

        ax[0].axis('off')
        ax[1].axis('off')

        #Show grayscale with annotations only
        ax2.imshow(rgb2gray(image),cmap=plt.cm.gray)
        if title:
            fig2.suptitle(title, fontsize=40)
        ax2.axis('off')


        #Check save status


        if save == 0 and savepath:
            makedir(savepath)
            path = os.path.join(savepath,title+'.png')
            fig.savefig(path)
        elif save == 1 and savepath:
            makedir(savepath)
            path = os.path.join(savepath,title+'.png')
            fig2.savefig(path)

        plt.show()





if __name__ == '__main__':
    #Paths
    folder_path = r'C:\Users\zagajewski\Desktop\AMR_ms_figures\SI - Example FoVs Clinical Isolates'
    im = r'48480_EC1_Untreated.tif'
    image_path = os.path.join(folder_path,im)

    segmenter_weights = r'C:\Users\zagajewski\Desktop\Deployment\mask_rcnn_EXP1.h5'
    classifier_weights = r'C:\Users\zagajewski\Desktop\AMR_ms_data_models\WT0CIP1_Holdout_Test\MODE - DenseNet121 BS - 16 LR - 0.0005 Holdout test.h5'

    #Mappings from training set
    map_resistant = {'colour': 'orangered', 'name': 'R'}
    map_sensitive = {'colour': 'dodgerblue', 'name': 'S'}
    mapping = {0:map_resistant, 1:map_sensitive}

    #Load image
    img = imread(image_path)


    results = segment_and_classify(img=img, segmenter=segmenter_weights, classifier=classifier_weights, filename='Sample')
    segmentations = results['segmentations']
    classifications = results['classifications']
    #Show results
    plot_detections(segmentations = segmentations,classifications=classifications,mappings =mapping, images=[img], show_caption=False, title=None)