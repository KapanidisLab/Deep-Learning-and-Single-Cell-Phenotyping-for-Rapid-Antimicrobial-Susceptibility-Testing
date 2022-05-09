
import os

import matplotlib.pyplot as plt

from ProcessingPipeline import ProcessingPipeline
from helpers import *
from skimage.io import imread
from segmentation import *
from classification import *
from keras.models import load_model
from Resistant_Sensitive_Comparison import amend_class_labels

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
    # Remove all edge detections
    segmentations, removed = remove_edge_cells(segmentations)
    print('Removed {} edge cells from image.'.format(removed))

    # Create and run classifier
    cells = apply_rois_to_image(input=segmentations, mode='masks', images=img)

    if cells == [[]]:
        print('No cells detected in image.')
        return [],[],[]

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

    return {'segmentations':segmentations, 'confidences':confidences, 'cells':cells, 'segmentations':segmentations, 'classifications':classifications}

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

    plt.legend(loc = 'upper left', fontsize=16)
    plt.title(title)
    plt.xlabel('Detection Confidence', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Normalised Frequency Density', fontsize=18)
    plt.tight_layout()
    plt.show()



def cells_vs_confidence(classifications = None, confidences = None, cells = None, title = None):

    #Resize to common shape
    cells = [pad_to_size(img, (64,64,3)) for img in cells]

    #Split by class
    untreated_idx = np.where(np.asarray(classifications) == 0)
    treated_idx = np.where(np.asarray(classifications) == 1)

    #Take treated susceptible as the axis target. Cells that are high in confidence in untreated are low in confidence in the treated susceptible.
    untreated_cells = np.asarray(cells)[untreated_idx]
    treated_cells = np.asarray(cells)[treated_idx]

    untreated_confidences = np.asarray(confidences)[untreated_idx]
    treated_confidences = np.asarray(confidences)[treated_idx]

    rescaled_untreated_confidences = np.abs((1.0 - untreated_confidences))

    #Put all cells and confidences together
    rescaled_all_confidences = np.append(treated_confidences, rescaled_untreated_confidences)
    all_cells = np.append(treated_cells, untreated_cells, axis=0)

    #Rebuild as lists
    rescaled_all_confidences = rescaled_all_confidences.tolist()
    all_cells_list = np.zeros(all_cells.shape[0]).tolist()
    for i,cell in enumerate(all_cells):
        all_cells_list[i] = cell

    #Sort
    sorted_confidences, sorted_cells = zip(*sorted(zip(rescaled_all_confidences, all_cells_list),key=lambda x: x[0]))

    sorted_confidences = np.asarray(sorted_confidences)
    sorted_cells = np.asarray(sorted_cells)

    #Prep display
    fig, axs = plt.subplots(5,20, constrained_layout=True, figsize=(20 * 2, 5 * 2))


    #Bin in intervals of 0.1
    thresholds = np.arange(0,1+0.05,0.05)
    for i in range(20): #Cycle through lower bounds
        j = i + 1 #Cycle through upper bounds

        lower_bound = thresholds[i]
        upper_bound = thresholds[j]

        print('Binning between {} - {}'.format(lower_bound,upper_bound))

        idx = np.where(np.logical_and(sorted_confidences>=lower_bound, sorted_confidences<=upper_bound))[0]
        print('Found {} cells'.format(len(idx)))
        cells_in_bin = sorted_cells[idx,:,:,:]

        #Select 5 random cells
        np.random.seed(42)
        selected_idx = np.random.choice(np.arange(0,cells_in_bin.shape[0],1), size=5, replace=False)
        selected_cells = cells_in_bin[selected_idx,:,:,:]

        for k in range(5):
            axs[k,i].imshow(skimage.img_as_ubyte(selected_cells[k,:,:,:]))
            plt.axis('off')
            axs[k,i].axis('off')

    plt.show()





if __name__ == '__main__':

    #Paths
    data_main = r'C:\Users\zagajewski\Desktop\Conor images new'
    speciesID = r'64017'
    repeatID = r'renamed'

    data_path = os.path.join(os.path.join(data_main,repeatID,speciesID))
    segmenter_weights = r'C:\Users\zagajewski\Desktop\Deployment\mask_rcnn_EXP1.h5'
    classifier_weights = r'C:\Users\zagajewski\Desktop\AMR_ms_data_models\WT0CIP1_Holdout_Test\MODE - DenseNet121 BS - 16 LR - 0.0005 Holdout test.h5'
    output = r'C:\Users\zagajewski\PycharmProjects\AMR\Data'


    cond_IDs = ['NA', 'CIP']
    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)

    map_WT = {'colour': 'orangered', 'name': 'Untreated'}
    map_CIP = {'colour': 'dodgerblue', 'name': 'CIP'}
    mapping = {0:map_WT, 1:map_CIP}


    #Make output structures
    output = os.path.join(output,'Classification_Distribution_'+speciesID)
    makedir(output)

    output_segregated = os.path.join(output,'Segregated')
    output_collected = os.path.join(output,'Collected')

    #Assemble images
    pipeline = ProcessingPipeline(data_path, 'NIM')
    #pipeline.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
     #             crop_mapping={'DAPI': 0, 'NR': 0},
      #            output_folder=output_segregated)
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

                    results = segment_and_classify(img=img, segmenter=segmenter,classifier=classifier,filename=file)
                    image_classifications = results['classifications']
                    image_confidences = results['confidences']
                    if image_classifications == []:
                        continue #Continue to next image if no cells detected

                    detection_count += len(image_classifications[0])

                    cumulative_classifications.extend(list(image_classifications[0]))
                    cumulative_confidences.extend(list(image_confidences[0]))

                    print('DONE {}'.format(image_count))

        #Plot histograms
        print('')
        print('Detected {} cells in {} images.'.format(detection_count,image_count))
        plot_distributions(classifications=cumulative_classifications,confidences=cumulative_confidences,mappings=mapping, title=speciesID+' '+cond_ID)





