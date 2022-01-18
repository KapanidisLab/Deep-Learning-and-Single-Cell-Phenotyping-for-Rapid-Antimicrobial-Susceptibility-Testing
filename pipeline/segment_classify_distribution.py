
import os

import matplotlib.pyplot as plt

from ProcessingPipeline import ProcessingPipeline
from helpers import *
from skimage.io import imread
from segmentation import *
from classification import *
from keras.models import load_model

def segment_and_classify(img=None, segmenter=None, classifier=None,filename=None):
    # Create an image for segmentation, fill 3 channels with NR
    img_NR = np.zeros(img.shape)
    img_NR[:, :, 0] = img[:, :, 0]
    img_NR[:, :, 1] = img[:, :, 0]
    img_NR[:, :, 2] = img[:, :, 0]

    # Expand to correct format
    img_NR = np.expand_dims(img_NR, axis=0)
    img = np.expand_dims(img, axis=0)

    # Create and run segmenter
    configuration = BacConfig()
    segmentations = predict_mrcnn_segmenter(source=img_NR, mode='images', weights=segmenter,
                                            config=configuration, filenames=filename)

    # Create and run classifier
    cells = apply_rois_to_image(input=segmentations, mode='masks', images=img)

    mean = np.asarray([0, 0, 0])
    resize_target = (64, 64, 3)

    # Go through all images
    classifications = []
    confidences = []
    for img_cells in cells:
        prediction,confidence, _ = predict(modelpath=classifier, X_test=img_cells, mean=mean,
                                    size_target=resize_target, pad_cells=True, resize_cells=False)
        classifications.append(prediction)
        confidences.append(confidence)

    return classifications,confidences

def plot_distributions(classifications=None, confidences=None, mappings=None, title=None):

    assert len(classifications) == len(confidences)
    total = len(classifications)


    classifications = np.asarray(classifications)
    confidences = np.asarray(confidences)

    cids = np.unique(classifications)
    print()
    print('Plotting distributions.')
    print('-----------------------------------')
    print('Detected {} classes: {}'.format(len(cids),cids))


    for cid in cids:

        idx_cid = np.where(classifications == cid,True,False)
        confidences_cid = confidences[idx_cid]

        #Compute histogram
        values_cid,bins = np.histogram(confidences_cid, bins=40, range=(0.5,1.0), density=False)

        #Normalise by total counts in all classes
        density_cid = values_cid / (total * np.diff(bins))

        #Plot using default utilties. Renormalise weight such that we're not rebinning the already binned histogram.

        colour = mappings[cid]['colour']
        name = mappings[cid]['name']
        plt.hist(bins[:-1], bins, weights=density_cid, edgecolor=colour, color=colour, histtype='stepfilled', alpha=0.2, label=name)



        proportion = np.sum(density_cid * np.diff(bins))
        print('Proportion in class {} labelled "{}" = {}'.format(cid,name,proportion))
        print('Minimum confidence = {}'.format(np.min(confidences_cid)))
        print('Maximum confidence = {}'.format(np.max(confidences_cid)))
        print('Number of detections = {}'.format(len(confidences_cid)))
        print('Plotting histogram...')

    plt.legend(loc = 'upper left')
    plt.title(title)
    plt.xlabel('Detection Confidence')
    plt.ylabel('Normalised Frequency Density')
    plt.show()





if __name__ == '__main__':

    #Paths
    speciesID = '48480'

    data_path = os.path.join(os.path.join(r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Clinical_strains\Repeat_0_10_11_21+Repeat_1_18_11_21',speciesID))
    segmenter_weights = r'C:\Users\zagajewski\Desktop\Deployment\mask_rcnn_EXP1.h5'
    classifier_weights = r'C:\Users\zagajewski\Desktop\Deployment\WT0_RIF1_holdout.h5'
    output = r'C:\Users\zagajewski\PycharmProjects\AMR\Data'


    cond_IDs = ['WT+ETOH', 'RIF+ETOH']
    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)

    map_WT = {'colour': 'orangered', 'name': 'WT'}
    map_CIP = {'colour': 'forestgreen', 'name': 'RIF'}
    mapping = {0:map_WT, 1:map_CIP}


    #Make output structures
    output = os.path.join(output,'Classification_Distribution_'+speciesID)
    makedir(output)

    output_segregated = os.path.join(output,'Segregated')
    output_collected = os.path.join(output,'Collected')

    #Assemble images
    pipeline = ProcessingPipeline(data_path, 'NIM')
    pipeline.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
                  crop_mapping={'DAPI': 0, 'NR': 0},
                  output_folder=output_segregated)
    pipeline.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected,
                     registration_target=0)


    #Load models
    print('LOADING CLASSIFIER...')
    classifier = load_model(classifier_weights)
    print('DONE \n')

    print('LOADING SEGMENTER...')
    configuration = BacConfig()
    configuration.IMAGES_PER_GPU = 1
    configuration.IMAGE_RESIZE_MODE = 'pad64' #Pad to multiples of 64
    configuration.__init__()

    segmenter = modellib.MaskRCNN(mode='inference', model_dir='../mrcnn/', config=configuration)
    segmenter.load_weights(segmenter_weights, by_name=True)
    print('DONE \n')

    #Loop over all conditions
    for cond_ID in cond_IDs:
        print('-----------------------')
        print('EVALUATING {}'.format(cond_ID))
        print('-----------------------')

        detection_count = 0
        image_count = 0

        cumulative_classifications = []
        cumulative_confidences = []

        #Find all images
        for root,dirs,files in os.walk(os.path.join(output_collected,cond_ID)):
            for file in files:
                if file.endswith('.tif'):

                    image_count += 1
                    img = imread(os.path.join(root,file))

                    image_classifications, image_confidences = segment_and_classify(img=img, segmenter=segmenter,classifier=classifier,filename=file)

                    detection_count += len(image_classifications[0])

                    cumulative_classifications.extend(list(image_classifications[0]))
                    cumulative_confidences.extend(list(image_confidences[0]))

                    print('DONE {}'.format(image_count))

        #Plot histograms
        print('')
        print('Detected {} cells in {} images.'.format(detection_count,image_count))
        plot_distributions(classifications=cumulative_classifications,confidences=cumulative_confidences,mappings=mapping, title=speciesID+' '+cond_ID)





