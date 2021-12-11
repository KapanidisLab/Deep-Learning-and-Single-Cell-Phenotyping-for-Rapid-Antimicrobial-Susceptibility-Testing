import matplotlib.pyplot as plt

from ProcessingPipeline import ProcessingPipeline as pipeline

import os
from pipeline.helpers import *
import numpy as np

import classification
import gc
from tensorflow.keras.backend import clear_session
from sklearn.metrics import ConfusionMatrixDisplay
from distutils.dir_util import copy_tree
import multiprocessing
import skimage.io
from tqdm import tqdm
from sklearn.utils import shuffle
from segmentation import *
from classification import *
from helpers import *
import os
from keras.models import load_model

def bayesian_simulation(list_of_classifications=None, mapping=None, cond_ID=None, max_cells=None, iterations=None):
    '''Run bayesian simulation on concetenated list of classifications'''

    def evaluate_likelihood(cumulative_cells=None, hypothesis_count=None):
        ids = np.arange(hypothesis_count)
        output = np.zeros(hypothesis_count)
        for i,id in enumerate(ids):

            #Count cells belonging to each class
            count = len(np.where(cumulative_cells==id)[0])

            #Evaluate likelihood
            likelihood = count/len(cumulative_cells)
            output[i] = likelihood

        return output

    def evaluate_posterior(likelihoods=None,priors=None, hypothesis_count=None):
        posteriors = likelihoods * priors / np.sum(likelihoods*priors)

        return posteriors






    #Concatenate all detections
    all_classifications = np.concatenate(list_of_classifications)
    assert len(all_classifications) >= max_cells, 'Found {} cells in {}, which is less than max_cells to include (max_cells ={})'.format(len(all_classifications),cond_ID,max_cells)

    hypothesis_count = len(mapping)


    posterior_evolution = np.zeros((hypothesis_count,max_cells,iterations))



    for i in tqdm(range(iterations), desc='Simulating {}'.format(cond_ID)):
        initial_priors = np.ones(3) / hypothesis_count

        step_prior = initial_priors

        #Shuffle
        shuffled_classifications = shuffle(all_classifications, random_state=i)

        for j in range(0,max_cells):
            cumulative_cells = shuffled_classifications[:j+1]

            step_likelihood = evaluate_likelihood(cumulative_cells = cumulative_cells, hypothesis_count=hypothesis_count)
            step_posterior = evaluate_posterior(likelihoods=step_likelihood,priors=step_prior,hypothesis_count=hypothesis_count)

            print('Posteriors after {} cells = {}'.format(j,step_posterior))

            #Store posterior
            posterior_evolution[:,j, i] = step_posterior

            #Update prior
            step_prior = step_posterior


    #Average over all iterations
    average_posteriors = np.mean(posterior_evolution,axis=2)

    #Plot
    for k in np.arange(hypothesis_count):
        posteriors = average_posteriors[k,:]

        colour = mapping[k]['colour']
        label = mapping[k]['name']

        plt.plot(posteriors, color = colour,label=label)

    plt.title('Bayesian simulation for condition {}'.format(cond_ID))
    plt.show()









def evaluate_bayesian_simulation(segmenter_weight_path=None, classifier_weight_path=None, experiment_path=None,output_path=None, size_target=None,
                              pad_cells=False, resize_cells=False,
                            cond_IDs=None, image_channels=None, img_dims =None,mapping=None,max_cells=None, iterations=None):
    # Make output folder
    makedir(output_path)

    # Fix pycharm console
    class PseudoTTY(object):
        def __init__(self, underlying):
            self.__underlying = underlying

        def __getattr__(self, name):
            return getattr(self.__underlying, name)

        def isatty(self):
            return True

    sys.stdout = PseudoTTY(sys.stdout)

    # Prepare train data
    output_segregated_test = os.path.join(output_path, 'Segregated_Test')
    output_collected_test = os.path.join(output_path, 'Collected_Test')

    pipeline_test = pipeline(experiment_path, 'NIM')
    pipeline_test.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
                             crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated_test)
    pipeline_test.path = output_segregated_test
    pipeline_test.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected_test,
                                registration_target=0)

    #Load classifying model
    classifying_model = load_model(classifier_weight_path)

    for cond_ID in cond_IDs:
        condition_path = os.path.join(output_collected_test,cond_ID)
        images = os.listdir(condition_path)

        images_for_segmentation = []
        images_multichannel = []

        for img in tqdm(images, desc='Preprocessing {}'.format(cond_ID)):
            image_path = os.path.join(condition_path,img)

            img = skimage.io.imread(image_path)

            # Create an image for segmentation, fill 3 channels with NR
            img_NR = np.zeros(img.shape)
            img_NR[:, :, 0] = img[:, :, 0]
            img_NR[:, :, 1] = img[:, :, 0]
            img_NR[:, :, 2] = img[:, :, 0]

            images_for_segmentation.append(img_NR)
            images_multichannel.append(img)

        # Create and run segmente
        configuration = BacConfig()
        print('Segmenting {}. This can take some time.'.format(cond_ID))
        segmentations = predict_mrcnn_segmenter(source=np.asarray(images_for_segmentation), mode='images', weights=segmenter_weight_path,
                                                config=configuration, filenames=images)

        #Cut out cells
        cells = apply_rois_to_image(input=segmentations, mode='masks', images=images_multichannel)

        mean = np.asarray([0, 0, 0])

        # Classify
        classifications = []
        for img_cells in tqdm(cells, desc='Classifying {}'.format(cond_ID)):
            prediction, model = predict(modelpath=classifying_model,X_test=img_cells, mean=mean, size_target=size_target, pad_cells=pad_cells, resize_cells=resize_cells)
            classifications.append(prediction)


        bayesian_simulation(list_of_classifications=classifications,mapping=mapping,cond_ID=cond_ID,max_cells=max_cells,iterations=iterations)


if __name__ == '__main__':

    output_path = os.path.join(get_parent_path(1), 'Data', 'LabStrains_Bayes')
    experiment_path = os.path.join(get_parent_path(1), 'Data', 'Exp1_HoldOut_Test','Repeat_7_01_12_21')
    cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
    image_channels = ['NR', 'DAPI']
    img_dims = (30, 684, 840)

    segmenter_weight_path = os.path.join(get_parent_path(1), 'firststage120211105T0215', 'mask_rcnn_EXP1.h5')
    classifier_weight_path = os.path.join(get_parent_path(1), 'Data','LabStrains_holdout', 'MODE - DenseNet121 BS - 16 LR - 0.0005 Holdout test.h5')

    annot_path = os.path.join(get_parent_path(1), 'Data', 'Segmentations_Edge_Removed')

    size_target = (64,64,3)

    max_cells = 50
    iterations = 25

    map_WT = {'colour': 'orangered', 'name': 'WT'}
    map_RIF = {'colour': 'dodgerblue', 'name': 'RIF'}
    map_CIP = {'colour': 'lawngreen', 'name': 'CIP'}
    mapping = {0:map_WT, 1:map_RIF, 2:map_CIP}

    evaluate_bayesian_simulation(segmenter_weight_path=segmenter_weight_path, classifier_weight_path=classifier_weight_path, experiment_path=experiment_path,output_path=output_path, size_target=size_target,
                              pad_cells=True, resize_cells=False,
                            cond_IDs=cond_IDs, image_channels=image_channels, img_dims =img_dims, mapping = mapping, max_cells=max_cells,iterations=iterations)



