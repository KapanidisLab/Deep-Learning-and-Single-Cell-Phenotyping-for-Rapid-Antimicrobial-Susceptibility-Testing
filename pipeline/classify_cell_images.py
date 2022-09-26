import os, skimage.io
from classification import predict
import numpy as np
from keras.models import load_model
import csv

if __name__ == '__main__':
    #Paths
    image_path = r'C:\Users\zagajewski\Desktop\For Alex model'
    classifier_weights = r'C:\Users\zagajewski\Desktop\AMR_ms_data_models\WT0CIP1_Holdout_Test\MODE - DenseNet121 BS - 16 LR - 0.0005 Holdout test.h5'
    output_path = r'C:\Users\zagajewski\Desktop\zooniverse_inference.csv'

    imglist = []
    fnames = []
    for root,dirs,files in os.walk(image_path):
        for file in files:
            if file.endswith('.jpg'):
                imglist.append(skimage.io.imread(os.path.join(root,file)))

                pre,suff = file.split('_')

                fnames.append(suff)

    print('Loaded {} images'.format(len(imglist)))

    mean = np.asarray([0, 0, 0])
    resize_target = (64, 64, 3)

    # Go through all images
    classifier = load_model(classifier_weights)
    print('Loaded model from {}, predicting...'.format(classifier_weights))

    predictions, confidences, _ = predict(modelpath=classifier, X_test=imglist, mean=mean,
                                        size_target=resize_target, pad_cells=True, resize_cells=False)

    print('Done, writing results...')

    with open(output_path, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        assert len(predictions) == len(confidences) == len(imglist) == len(fnames)

        for i in range(len(predictions)):
            row = [fnames[i],predictions[i],confidences[i]]
            writer.writerow(row)

    print('Done! :)')

