import keras.callbacks
import sklearn.utils.multiclass
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras import layers, activations, losses, optimizers, metrics, Model, callbacks
from keras.optimizers import SGD,Adam, Nadam, Adagrad
from keras.regularizers import l2
import numpy as np
import sklearn, skimage
import sklearn.decomposition, sklearn.discriminant_analysis
from classification import pad_to_size
from imgaug import augmenters as iaa
from Datagen_imgaug import DataGenerator
from helpers import summarize_triplet_loss
import os
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

import seaborn


from helpers import *

import itertools, multiprocessing

from triplet_loss import triplet_hard_loss, triplet_semihard_loss, pairwise_distance

class Visualize_Embeddings_LDA(keras.callbacks.Callback):
    '''Visualise LDA'd embeddings at given epoch starts. Useful for tracking model learning.'''
    def __init__(self, *args, **kwargs):
        self.validation_X = kwargs.pop('validation_X')
        self.validation_y = kwargs.pop('validation_y')
        self.target_epochs = kwargs.pop('target_epochs')
        self.class_mapping = kwargs.pop('class_mapping')

        super(Visualize_Embeddings_LDA, self).__init__(*args, **kwargs)

    def on_train_end(self, logs=None):
        self.visualise_embeddings('final')

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.target_epochs:
            self.visualise_embeddings(epoch)

    def visualise_embeddings(self, epoch):
        embeddings = self.model.predict(self.validation_X)
        components = self.LDA_embeddings(embeddings, self.validation_y)

        plot_title = '1-component LDA at epoch {}'.format(str(epoch))
        self.LDA_visualise(components,self.validation_y, self.class_mapping, plot_title=plot_title)

    @staticmethod
    def LDA_embeddings(embeddings,labels):
        LDA = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)

        components = LDA.fit_transform(X=embeddings,y=labels)
        return components

    @staticmethod
    def LDA_visualise(components, labels, class_mapping, plot_title=None):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        names = []
        for class_id, mapping in class_mapping.items():
            idx = labels == class_id

            matching_components = components[idx]

            colour = mapping['colour']
            ax.hist(matching_components,bins=100,edgecolor=colour, color=colour, histtype='stepfilled', alpha=0.2)
            ax.set_xlabel('LDA 1', fontsize=15)

            names.append(mapping['name'])


        if plot_title is not None:
            ax.set_title(plot_title, fontsize=20)

        ax.legend(names)
        plt.show()



class Visualize_Embeddings_PCA(keras.callbacks.Callback):
    '''Visualise PCA'd embeddings at given epoch starts. Useful for tracking model learning.'''

    def __init__(self, *args, **kwargs):
        self.validation_X = kwargs.pop('validation_X')
        self.validation_y = kwargs.pop('validation_y')
        self.target_epochs = kwargs.pop('target_epochs')
        self.class_mapping = kwargs.pop('class_mapping')

        super(Visualize_Embeddings_PCA, self).__init__(*args, **kwargs)

    def on_train_end(self, logs=None):
        self.visualise_embeddings('final')

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.target_epochs:
            self.visualise_embeddings(epoch)


    def visualise_embeddings(self,epoch):
        embeddings = self.model.predict(self.validation_X)
        components, variance_ratio = self.PCA_embeddings(embeddings)

        plot_title = '2-component PCA at epoch {}'.format(str(epoch))
        self.PCA_visualise(self.validation_y, components, variance_ratio, self.class_mapping, plot_title=plot_title)


    @staticmethod
    def PCA_embeddings(embeddings):
        StandardScaler = sklearn.preprocessing.StandardScaler()
        embeddings = StandardScaler.fit_transform(embeddings)
        PCA = sklearn.decomposition.PCA(n_components=2)

        components = PCA.fit_transform(embeddings)
        explained_varience = PCA.explained_variance_ratio_
        return components, explained_varience

    @staticmethod
    def PCA_visualise(labels,components,explained_variance, class_mapping, plot_title=None):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        names=[]
        for class_id,mapping in class_mapping.items():
            idx = labels == class_id

            matching_components_1 = components[idx,0]
            matching_components_2 = components[idx,1]

            ax.scatter(matching_components_1,matching_components_2,c=mapping['colour'],s=1)

            names.append(mapping['name'])

        ax.set_xlabel('Principal Component 1 : {:.2f} total variance'.format(np.around(explained_variance[0],2)), fontsize=15)
        ax.set_ylabel('Principal Component 2 : {:.2f} total variance'.format(np.around(explained_variance[1],2)), fontsize=15)

        if plot_title is not None:
            ax.set_title(plot_title, fontsize=20)

        ax.legend(names)
        plt.show()

@tf.function
def _calc_average_vector(y_true,y_pred,cid):
    idx = tf.where(tf.equal(y_true,cid))
    vecs = tf.gather(y_pred,
            indices=idx)
    return tf.reduce_mean(vecs,0)


@tf.function
def WT2WT_mean(y_true,y_pred):
    avg_WT = _calc_average_vector(y_true, y_pred, 0)

    idx = tf.where(tf.equal(y_true,0))
    vecs = tf.gather(y_pred,
            indices=idx)

    norms = tf.norm(tf.subtract(vecs,avg_WT), axis=-1)
    return tf.reduce_mean(norms)

@tf.function
def CIP2CIP_mean(y_true,y_pred):
    avg_CIP = _calc_average_vector(y_true, y_pred, 1)

    idx = tf.where(tf.equal(y_true,1))
    vecs = tf.gather(y_pred,
            indices=idx)

    norms = tf.norm(tf.subtract(vecs,avg_CIP), axis=-1)
    return tf.reduce_mean(norms)

@tf.function
def mean2mean(y_true,y_pred):
    '''y_true are the class labels 1D numpy vector, y_pred are the embeddings (nsamples,nfeatures)'''

    avg_0 = _calc_average_vector(y_true, y_pred, 0)
    avg_1 = _calc_average_vector(y_true, y_pred, 1)

    return tf.norm(tf.subtract(avg_1,avg_0), axis=-1)

@tf.function
def WT2CIP_mean(y_true, y_pred):
    avg_CIP = _calc_average_vector(y_true, y_pred, 1)

    idx = tf.where(tf.equal(y_true,0))
    vecs = tf.gather(y_pred,
            indices=idx)

    norms = tf.norm(tf.subtract(vecs,avg_CIP), axis=-1)
    return tf.reduce_mean(norms)

@tf.function
def CIP2WT_mean(y_true, y_pred):
    avg_WT = _calc_average_vector(y_true, y_pred, 0)

    idx = tf.where(tf.equal(y_true,1))
    vecs = tf.gather(y_pred,
            indices=idx)

    norms = tf.norm(tf.subtract(vecs,avg_WT), axis=-1)
    return tf.reduce_mean(norms)


def LR_Schedule(epoch,lr,total_epochs=None, initial_lr=None):

    if epoch < round(0.8*total_epochs):
        return lr
    else:
        return initial_lr/ 10  # initial learning rate

def version1(input,embedding_size):
    'Base case - dense cascade with BN'
    dense1 = layers.Dense(512,kernel_regularizer=l2(0.01))(input)
    dense1_relu = layers.Activation(activations.relu)(dense1)
    dense1_BN = layers.BatchNormalization()(dense1_relu)

    dense2 = layers.Dense(256,kernel_regularizer=l2(0.01))(dense1_BN)
    dense2_relu = layers.Activation(activations.relu)(dense2)
    dense2_BN = layers.BatchNormalization()(dense2_relu)

    output = layers.Dense(embedding_size,kernel_regularizer=l2(0.01))(dense2_BN)
    output_normal = layers.Lambda(lambda x: K.l2_normalize(x,axis=-1))(output)

    print('Using encoder version 1, embedding size {}'.format(embedding_size))
    return output_normal

def version0(input, embedding_size):
    dense1 = layers.Dense(512)(input)
    dense1_relu = layers.Activation(activations.relu)(dense1)

    dense2 = layers.Dense(256)(dense1_relu)
    dense2_relu = layers.Activation(activations.relu)(dense2)

    dense3 = layers.Dense(embedding_size)(dense2_relu)
    output_normal = layers.Lambda(lambda x: K.l2_normalize(x,axis=-1))(dense3)

    print('Using encoder version 0, embedding size {}'.format(embedding_size))
    return output_normal


def define_distance_model(backbone_weights=None, target_shape=None,encoder_version=None, optimizer=None, initial_lr=None, embedding_size=None):


    # Create backbone with defined weights
    print('Defining DenseNet121...')
    model = DenseNet121(include_top=False, weights=None, input_shape=target_shape)
    if backbone_weights == None:
        print('Random weight initialization...')
    else:
        print('Loading provided weights by name...')
        model.load_weights(backbone_weights,by_name=True)

    freezeable_layers = [layer for layer in model.layers]

    for layer in model.layers:
        layer.trainable = True


    #Pool final feature map
    avgpool2d = layers.GlobalAvgPool2D()(model.output)

    #Add encoder according to specification
    if encoder_version == 1:
        output = version1(avgpool2d,embedding_size)
    elif encoder_version ==0:
        output = version0(avgpool2d,embedding_size)
    else:
        raise ValueError('Version parameter contains an unsupported value.')

    # Assemble encoder
    similarity_model = Model(model.input, output, name="Similarity_model")


    #Compile with optimizer and triplet semihard loss
    # Select optimimzer
    if optimizer == 'SGD+N':  # SGD with nestrov
        opt = SGD(lr=initial_lr, momentum=0.9, nesterov=True)  # SGD with nesterov momentum, no vanilla version
    elif optimizer == 'SGD':  # SGD with ordinary momentum
        opt = SGD(lr=initial_lr, momentum=0.9, nesterov=False)  # SGD with nesterov momentum, no vanilla version
    elif optimizer == 'NAdam':
        opt = Nadam(lr=initial_lr)  # Nestrov Adam
    elif optimizer == 'Adam':
        opt = Adam(lr=initial_lr)  # Adam
    elif optimizer == 'Adagrad':
        opt = Adagrad(lr=initial_lr)
    else:
        raise TypeError('Optimizer {} not supported'.format(optimizer))

    #Compile first with semihard loss
    similarity_model.compile(optimizer=opt, loss=triplet_semihard_loss, metrics=[triplet_semihard_loss,triplet_hard_loss,mean2mean,WT2CIP_mean,CIP2WT_mean, WT2WT_mean, CIP2CIP_mean])

    keras.utils.plot_model(
        similarity_model, to_file=r'C:\Users\zagajewski\Desktop\encoder.png',show_shapes=True)

    return similarity_model, freezeable_layers



def create_similarity_generator(cells=None, labels=None, batch_size=None, aug1=None,aug2=None):

    #Shuffle
    images_shuffled, labels_shuffled = sklearn.utils.shuffle(cells,labels, random_state=42)

    #Split off test set
    images_train_val, images_test, labels_train_val, labels_test = sklearn.model_selection.train_test_split(images_shuffled, labels_shuffled, test_size = 0.2, random_state = 42)

    #Split remainder into training and validation
    images_train, images_val, labels_train, labels_val = sklearn.model_selection.train_test_split(images_train_val,labels_train_val, test_size=0.2, random_state=42)

    if aug1 == None and aug2 == None:
        augment = False
    else:
        augment = True

    traingen = DataGenerator(images_train, labels_train,
                             batch_size=batch_size, shuffle=True, augment=augment, aug1=aug1, aug2=aug2)

    return traingen, (images_val,labels_val), (images_test,labels_test), (images_train,labels_train)



def train_triplet_similarity(backbone_weights=None, cells=None, labels=None, target_shape=None, encoder_version=None, optimizer=None, initial_lr=None, batch_size=None, logdir=None, dt_string=None, embedding_size=None, epochs=None):

    # Fix pycharm console
    class PseudoTTY(object):
        def __init__(self, underlying):
            self.__underlying = underlying

        def __getattr__(self, name):
            return getattr(self.__underlying, name)

        def isatty(self):
            return True

    sys.stdout = PseudoTTY(sys.stdout)

    #Preprocess images
    print('Padding cell images to {}'.format(target_shape))
    cells = [pad_to_size(img,target_shape) for img in cells]

    cells = skimage.img_as_ubyte(np.asarray(cells))
    labels = np.asarray(labels,dtype='int32')
    #Equalize all histograms
    histeq = iaa.Sequential([iaa.AllChannelsHistogramEqualization()])
    cells = histeq(images = cells)


    #Create model
    similarity_model,freezable_layers = define_distance_model(backbone_weights=backbone_weights, target_shape=target_shape, encoder_version=encoder_version,optimizer=optimizer,initial_lr=initial_lr, embedding_size=embedding_size)

    #Define augmentation

    seq1 = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                rotate=(-90, 90),
                mode="constant",
                cval=0
            ),
            iaa.Sometimes(0.5, iaa.ShearY(shear=(-45, 45))),
            iaa.Sometimes(0.5, iaa.ShearX(shear=(-45, 45)))
        ],
        random_order=True)

    seq2 = iaa.Sequential(
        [
            iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5))),  # Random sharpness increae
            iaa.Sometimes(0.5, iaa.WithChannels(0, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}))),
            # Random up to 10% misalignment
            iaa.Sometimes(0.5, iaa.MultiplyBrightness((0.5, 2.0))),  # Brightness correction
            iaa.Sometimes(0.5, iaa.imgcorruptlike.GaussianNoise(severity=(1, 2))),  # Random gaussian noise
            # iaa.Sometimes(0.5, iaa.imgcorruptlike.DefocusBlur(severity=(1,2))), #Defocus correction
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.5)))
        ],
    )

    #Create dataset
    traingen, (val_X,val_y), (test_X,test_y), (train_X,train_y) = create_similarity_generator(cells=cells,labels=labels, batch_size=batch_size, aug1=seq1, aug2=seq2)

    #Callbacks
    checkpoint_name = dt_string+'.h5'
    class_mapping = {0:{'name':'WT', 'colour':'red'}, 1:{'name':'CIP', 'colour':'blue'}}

    cbacks = [
        callbacks.TensorBoard(log_dir=logdir,
                                    histogram_freq=0, write_graph=False, write_images=False),
        callbacks.ModelCheckpoint(os.path.join(logdir,checkpoint_name),
                                        verbose=0, save_weights_only=False, save_best_only=True, monitor='loss',
                                        mode='min'),
        #Visualize_Embeddings_LDA(validation_X=val_X, validation_y=val_y, target_epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], class_mapping = class_mapping),
        Visualize_Embeddings_PCA(validation_X=val_X, validation_y=val_y, target_epochs= [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                                 class_mapping=class_mapping),
        callbacks.LearningRateScheduler(schedule= lambda epoch,lr,initial_lr=initial_lr, total_epochs=epochs : LR_Schedule(epoch,lr,total_epochs=total_epochs,initial_lr=initial_lr),verbose=0)

    ]
    #Train semihard loss first
    history_semihard = similarity_model.fit_generator(traingen, steps_per_epoch=len(traingen), validation_data=(val_X,val_y),  epochs=epochs, callbacks=cbacks)

    #Recompile with hard loss and finetune
    similarity_model.compile(optimizer=similarity_model.optimizer, loss=triplet_hard_loss, metrics=similarity_model.metrics)

    history_hard = similarity_model.fit_generator(traingen, steps_per_epoch=len(traingen), validation_data=(val_X, val_y),
                                             epochs=epochs+10, callbacks=cbacks, initial_epoch=epochs)

    #Concatanate histories
    history_total = {}
    for key,value in history_semihard.history.items():
        history_total[key] = history_semihard.history[key] + history_hard.history[key]

    #Plot basic stats
    summarize_triplet_loss(history_total, checkpoint_name)

def optimize_triplet_similarity(backbone_weights = None, cells = None, labels = None, target_shape = None, encoder_version = None, parameter_grid = None, logdir = None):

    # Compute permutations of main parameters, call train() for each permutation. Wrap in multiprocessing to force GPU
    # memory release between runs, which otherwise doesn't happen

    # Fix pycharm console
    class PseudoTTY(object):
        def __init__(self, underlying):
            self.__underlying = underlying

        def __getattr__(self, name):
            return getattr(self.__underlying, name)

        def isatty(self):
            return True

    sys.stdout = PseudoTTY(sys.stdout)

    keysum = ['batch_size', 'learning_rate', 'epochs', 'optimizer', 'embedding_size']
    assert all([var in parameter_grid for var in keysum]), 'Check all parameters given'

    makedir(logdir)  # Creat dir for logs

    for i, permutation in enumerate(itertools.product(parameter_grid['batch_size'], parameter_grid['learning_rate'],
                                                      parameter_grid['epochs'], parameter_grid['optimizer'], parameter_grid['embedding_size'])):

        (batch_size, learning_rate, epochs, optimizer, embedding_size) = permutation  # Fetch parameters
        dt_string = "Encoder V{} BS {}, LR {}, epochs {}, opt {}, embedding_size {}".format(encoder_version, batch_size, learning_rate, epochs, optimizer, embedding_size)

        # Create separate subdir for each run, for tensorboard ease

        logdir_run = os.path.join(logdir, dt_string)
        makedir(logdir_run)

        kwargs = {'backbone_weights':backbone_weights, 'cells':cells, 'labels':labels, 'target_shape':target_shape, 'encoder_version':encoder_version, 'optimizer':optimizer, 'initial_lr':learning_rate, 'batch_size':batch_size, 'logdir':logdir_run, 'dt_string':dt_string, 'embedding_size':embedding_size, 'epochs':epochs }

        p = multiprocessing.Process(target=train_triplet_similarity, kwargs=kwargs)
        p.start()
        p.join()


def visualise_distance_matrices(complete=None, complete_separator=None):

    #Plot joint heatmap first
    fig,ax = plt.subplots(1,1, figsize=(10,10))
    seaborn.heatmap(complete, cmap='inferno', cbar=True, ax=ax, square=True)

    ax.invert_yaxis()
    ax.set_title('Complete distance matrix')
    plt.show()



def compute_distance_matrices(untreated_embeddings=None,treated_embeddings=None):

    count_untreated = len(untreated_embeddings)
    count_treated = len(treated_embeddings)

    complete = tf.convert_to_tensor(np.concatenate([untreated_embeddings,treated_embeddings], axis=0))

    #Compute joint
    complete = pairwise_distance(complete,squared=False)

    return complete.eval(), count_untreated


def evaluate_similarity(model = None, untreated_cells=None, treated_cells=None, target_shape = None):
    #Pick out treated and untreated cells

    # Fix pycharm console
    class PseudoTTY(object):
        def __init__(self, underlying):
            self.__underlying = underlying

        def __getattr__(self, name):
            return getattr(self.__underlying, name)

        def isatty(self):
            return True

    sys.stdout = PseudoTTY(sys.stdout)

    #Preprocess images
    print('Padding cell images to {}'.format(target_shape))
    treated_cells = [pad_to_size(img,target_shape) for img in treated_cells]
    untreated_cells = [pad_to_size(img,target_shape) for img in untreated_cells]

    treated_cells = skimage.img_as_ubyte(np.asarray(treated_cells))
    untreated_cells = skimage.img_as_ubyte(np.asarray(untreated_cells))


    #Equalize all histograms
    histeq = iaa.Sequential([iaa.AllChannelsHistogramEqualization()])
    untreated_cells = histeq(images = untreated_cells)
    treated_cells = histeq(images = treated_cells)

    #Pick out treated and untreated cells
    #Generate embeddings
    print('Generating treated embeddings.')
    treated_latent = model.predict(treated_cells)

    print('Generating untreated embeddings.')
    untreated_latent = model.predict(untreated_cells)

    print('Done')

    untreated_labels = np.zeros(len(untreated_latent))
    treated_labels = np.ones(len(treated_latent))
    class_mapping = {0:{'name':'WT', 'colour':'red'}, 1:{'name':'CIP', 'colour':'blue'}}
    complete_latent = np.concatenate([untreated_latent,treated_latent],axis=0)
    complete_labels = np.concatenate([untreated_labels,treated_labels],axis=0)


    components, explained_varience = Visualize_Embeddings_PCA.PCA_embeddings(complete_latent)
    Visualize_Embeddings_PCA.PCA_visualise(complete_labels,components,explained_varience, class_mapping, plot_title='2-component PCA')

    complete, separator = compute_distance_matrices(untreated_embeddings=untreated_latent, treated_embeddings=treated_latent)
    visualise_distance_matrices(complete=complete, complete_separator=separator)





