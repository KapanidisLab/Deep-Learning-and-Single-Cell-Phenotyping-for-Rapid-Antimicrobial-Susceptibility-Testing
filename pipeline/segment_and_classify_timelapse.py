import matplotlib.pyplot as plt
from skimage.io import imread
from segmentation import *
from helpers import *

from keras.models import load_model
from segment_classify_distribution import segment_and_classify
from segment_and_classify import plot_detections

def plot_detection_timelapse(classifications=None,delta_t=None,timeunit=None,mapping=None,title=None):

    tpoints = len(classifications) #number of different timepoints
    ratios = {}

    #Calculate ratios of all classes
    for key,item in mapping.items():
        ratios[key] = np.zeros(tpoints)
        for i,tpoint in enumerate(classifications):
            ratio = (tpoint == key).sum()/(len(tpoint))
            ratios[key][i] = ratio

    x = np.arange(0,tpoints*delta_t,delta_t)

    #Plot on graph
    for key,item in ratios.items():

        colour = mapping[key]['colour']
        name = mapping[key]['name']
        plt.plot(x,item,color=colour, marker='o', label=name)

    if title:
        plt.title(title,fontsize=16)

    plt.legend(fontsize=16)
    plt.xlabel('Time ({})'.format(timeunit), fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Class detection ratio', fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_individual_cells(classifications=None,delta_t=None,timeunit=None,mapping=None, n=[0,2,3]):
    pass

if __name__ == '__main__':

    image_path = os.path.join(r'C:\Users\zagajewski\Desktop\Timelapses\Processed')
    filename = 'COAMOX_timelapse.tif'

    segmenter_weights = r'C:\Users\zagajewski\Desktop\Deployment\mask_rcnn_EXP1.h5'
    classifier_weights = r'C:\Users\zagajewski\Desktop\AMR_ms_data_models\WT0COAMOX1_Holdout_Test\MODE - DenseNet121 BS - 8 LR - 0.001 Holdout test.h5'

    #Mappings from training set
    map_resistant = {'colour': 'orangered', 'name': 'Untreated'}
    map_sensitive = {'colour': 'dodgerblue', 'name': 'CIP'}
    mapping = {0:map_resistant, 1:map_sensitive}

    #Load image
    img = imread(os.path.join(image_path,filename))
    img = np.asarray(img,dtype='uint16')
    print(img.dtype)
    print(img.shape)

    img_count = img.shape[0]
    classifaction_stack = []

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

    for i in range(img_count):
        print('Processing {}'.format(i))
        image = img[i,:,:,:]
        image.dtype
        dtype = img.dtype

        #Reshape image

        if image.shape[0] == 2: #Add extra 0 blue channel if not present
            zero = np.zeros((image.shape[1],image.shape[2],3),dtype=dtype)
            for ch in range(image.shape[0]):
                zero[:,:,ch] = image[ch,:,:]
            image = zero


        results = segment_and_classify(img=image, segmenter=segmenter, classifier=classifier,
                                       filename='t{}'.format(i))
        print('Processed timepoint {}...'.format(i))

        segmentations = results['segmentations']
        classifications = results['classifications']

        classifaction_stack.append(classifications[0])

        title = 't={}'.format(i)
         #Show results
        #plot_detections(segmentations=segmentations, classifications=classifications, mappings=mapping, images=[image],
         #              show_caption=False, title=title, save=0, savepath=image_path)

    plot_detection_timelapse(classifications=classifaction_stack,delta_t=2,timeunit='min',mapping=mapping, title='+COAMOX')