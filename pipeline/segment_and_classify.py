

if __name__ == '__main__':

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

    def plot_detections(segmentations=None,classifications=None,images=None, mappings=None, show_caption=True, title=''):

        assert len(segmentations) == len(classifications)

        for i,seg in enumerate(segmentations):

            fig, ax = plt.subplots(1,2, figsize=(16,16))
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

                #Caption
                if show_caption:
                    caption = "{}".format(name)
                    ax[0].text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="none")

                #Plot original image unchanged, and beside it grayscaled and with annotations
                image = img_as_ubyte(image) #8bit conversion
                ax[1].imshow(image)
                ax[0].imshow(rgb2gray(image),cmap=plt.cm.gray)

            plt.title(str(title),fontsize=22)
            plt.show()

    #Paths
    filename = "210403_1_AMR_combined_3_CIP+ETOH_posXY0.tif"
    image_path = os.path.join(get_parent_path(1), "Data" , "Test_4_multichannel_collected", "CIP+ETOH", filename)
    segmenter_weights = os.path.join(get_parent_path(1), "predconfig_1+320211013T2353","mask_rcnn_predconfig_1+3.h5")
    classifier_weights = os.path.join(get_parent_path(1),"predconfig_1+320211013T2353","DenseNet121 BS 16, LR 0.0005, epochs 100, opt NAdam.h5")

    #Mappings from training set
    map_WT = {'colour': 'orangered', 'name': 'WT'}
    map_RIF = {'colour': 'dodgerblue', 'name': 'RIF'}
    map_CIP = {'colour': 'lawngreen', 'name': 'CIP'}
    mapping = {0:map_WT, 1:map_RIF, 2:map_CIP}


    #Load image
    img = imread(image_path)

    #Create an image for segmentation, fill 3 channels with NR
    img_NR = np.zeros(img.shape)
    img_NR[:,:,0] = img[:,:,0]
    img_NR[:,:,1] = img[:,:,0]
    img_NR[:,:,2] = img[:,:,0]

    #Expand to correct format
    img_NR = np.expand_dims(img_NR, axis=0)
    img = np.expand_dims(img, axis=0)

    #Create and run segmenter
    configuration = BacConfig()
    segmentations = predict_mrcnn_segmenter(source=img_NR, mode='images', weights=segmenter_weights, config=configuration, filenames=filename)

    #Create and run classifier
    cells = apply_rois_to_image(input=segmentations, mode='masks', images=img)

    mean = np.asarray([0, 0, 0])
    resize_target = (64,64,3)

    #Go through all images
    classifications =[]
    for img_cells in cells:
        prediction,model = predict(modelpath=classifier_weights, X_test=img_cells, mean=mean, resize_target=resize_target)
        classifications.append(prediction)

    #Show results
    plot_detections(segmentations = segmentations,classifications=classifications,mappings =mapping, images=img, show_caption=True, title=filename)