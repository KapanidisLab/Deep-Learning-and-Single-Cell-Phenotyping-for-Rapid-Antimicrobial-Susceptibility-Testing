import os

import matplotlib.pyplot as plt

from ProcessingPipeline import ProcessingPipeline
from helpers import *
from skimage.io import imread
from segmentation import *
from classification import *
from keras.models import load_model
from Resistant_Sensitive_Comparison import amend_class_labels

from segment_classify_distribution import segment_and_classify, plot_distributions, cells_vs_confidence

def evaluate_titratiion(data_folder = None, segmenter_weights = None, classifier_weights = None, output = None, cond_IDs = None, image_channels = None, img_dims = None, mapping=None):

    makedir(output)
    #Identify concentration points
    data_paths = []
    concentrations = []
    for concentration in os.listdir(data_folder):

        #Strip square brackets, insert decimal
        conc = concentration
        if conc.startswith('0'):
            conc = conc[:1] + '.' + conc[1:]
        elif len(conc) == 1:
            conc = conc + '.0'

        #Cast to float
        conc = float(conc)

        print('Detected concentration point = {}'.format(conc))
        data_paths.append(os.path.join(data_folder,concentration))
        concentrations.append(conc)

    # Load models
    print('LOADING CLASSIFIER...')
    classifier = load_model(classifier_weights)
    print('DONE \n')

    print('LOADING SEGMENTER...')
    configuration = BacConfig()
    configuration.IMAGES_PER_GPU = 1
    configuration.IMAGE_RESIZE_MODE = 'pad64'  # Pad to multiples of 64
    configuration.__init__()

    segmenter = modellib.MaskRCNN(mode='inference', model_dir='../mrcnn/', config=configuration)
    segmenter.load_weights(segmenter_weights, by_name=True)
    print('DONE \n')

    #Evaluate
    for i,path in enumerate(data_paths):
        conc = concentrations[i]
        print('Evaluating {}'.format(path))

        # Make output structures
        output_conc = os.path.join(output, 'Concentration {}'.format(conc))
        makedir(output_conc)

        output_segregated = os.path.join(output_conc, 'Segregated')
        output_collected = os.path.join(output_conc, 'Collected')

        # Assemble images
        pipeline = ProcessingPipeline(path, 'NIM')
        pipeline.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
                      crop_mapping={'DAPI': 0, 'NR': 0},
                      output_folder=output_segregated)
        pipeline.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected,
                         registration_target=0)

        # Loop over all conditions
        for cond_ID in cond_IDs:

            detection_count = 0
            image_count = 0

            cumulative_classifications = []
            cumulative_confidences = []
            cumulative_cells = []
            cumulative_segmentations = []

            # Find all images
            for root, dirs, files in os.walk(os.path.join(output_collected, cond_ID)):
                for file in files:
                    if file.endswith('.tif'):
                        image_count += 1
                        img = imread(os.path.join(root, file))

                        results = segment_and_classify(img=img, segmenter=segmenter, classifier=classifier, filename=file)

                        image_classifications = results['classifications']
                        image_confidences = results['confidences']
                        image_cells = results['cells']

                        detection_count += len(image_classifications[0])

                        cumulative_classifications.extend(list(image_classifications[0]))
                        cumulative_confidences.extend(list(image_confidences[0]))
                        cumulative_cells.extend(list(image_cells[0]))

                        print('DONE {}'.format(image_count))

            # Plot histograms
            print('')
            print('Detected {} cells in {} images.'.format(detection_count, image_count))
            plot_distributions(classifications=cumulative_classifications, confidences=cumulative_confidences,
                               mappings=mapping, title='{} concentration {}'.format(cond_ID,conc))

            # Plot cells as a function of confidence
            cells_vs_confidence(classifications = cumulative_classifications, confidences = cumulative_confidences, cells = cumulative_cells, title='{} confidence at concentration {}'.format(cond_ID,conc))



if __name__ == '__main__':

    output = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\titration'
    segmenter_weights = r'C:\Users\zagajewski\Desktop\Deployment\mask_rcnn_EXP1.h5'
    classifier_weights = r'C:\Users\zagajewski\Desktop\AMR_ms_data_models\WT0CIP1_Holdout_Test\MODE - DenseNet121 BS - 16 LR - 0.0005 Holdout test.h5'

    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)
    cond_IDs = ['CIP+ETOH']

    map_WT = {'colour': 'orangered', 'name': 'Untreated'}
    map_CIP = {'colour': 'dodgerblue', 'name': 'CIP'}
    mapping = {0:map_WT, 1:map_CIP}

    data_folder = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\23_03_22'

    evaluate_titratiion(data_folder = data_folder, segmenter_weights = segmenter_weights, classifier_weights = classifier_weights, output = output, cond_IDs = cond_IDs, image_channels = image_channels, img_dims =img_dims, mapping=mapping)