# importing all the required packages

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical 

class mnist_dnn():
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.results = dict()
        self.df_results = None
        # the data and the model have been initiated.
    def dnn_output(self, layers, neurons, activations, epochs):
        '''
        This function intends to train a model, fit it to the training data
        and then evaluate the model on the testing data. The input parameters,
        layers -> Number of hidden layers in the model.
        neurons -> number of neurons in each layer
        activations -> The activations functions of each layer
        epochs -> The number of epochs

        What the function returns:
        [<message>, <error_code>]
        0 -> Successfully compiled
        -1 -> Lengths of neurons and activations don't match
        '''        
        if(len(neurons) != len(activations) or len(neurons) != layers or len(activations) != layers):
            return "Error", -1
        model = tf.keras.models.Sequential(name='dnn_mnist')
        model.add(Flatten(input_shape=(28,28)))
        for i in range((layers)): # now we shall iterate through each of the layers
            # thus keep on builiding the model
            model.add(Dense(neurons[i], activation = activations[i]))
        model.add(Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
        # fitting the training data
        model.fit(self.x_train, self.y_train, epochs=epochs, verbose = 0)
        train_results = model.evaluate(self.x_train, self.y_train, verbose=0)
        test_results = model.evaluate(self.x_test, self.y_test, verbose=0)
        self.results = {'test_results': test_results, 'train_results':train_results}
        return "Model Successfully Compiled", 0
    def check_results(self, res='All'):
        '''
        This function is used to print the results of test or train or both
        datasets on the most recently compiled NN model. 
        '''
        if(self.results):
            df = pd.DataFrame(self.results, index=['Loss', 'Percent Accuracy'])
            self.df_results = df
            print(df)
        else:
            return "Model Not Compiled"


class mnist_cnn():
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = np.reshape(self.x_train, (-1, 28, 28, 1)).astype('float32')
        self.x_test = np.reshape(self.x_test, (-1, 28, 28, 1)).astype('float32')
        self.y_test = to_categorical(self.y_test, num_classes = 10)
        self.y_train = to_categorical(self.y_train, num_classes = 10)
        self.normalize = False
        self.results = dict()
    def normalize_data(self):
        '''
        This function enables the user to Normalize the data on one's wish
        One may try to use the normal data, but in case the user needs the normalization
        he/she can just invoke this function and proceed with all the downstream tasks
        '''
        if(self.normalize == False):
            self.x_train = self.x_train/255
            self.x_test = self.x_test/255
            self.normalize = True
            return "Successfully Normalized", 0
        return "Data Already Normalzed", -1
    def cnn_output(self, epochs=5, verbose = 0):
        '''
        This function uses a fixed architcture of neural networks, which has known to perform better
        the user can modify the number of epochs. 
        '''
        model = Sequential()
        # the input layer
        model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(28,28,1)))
        # the output of this layer is (28-5+1)X(28-5+1) = 24X24 Images
        model.add(MaxPooling2D(padding="same"))
        # the output of this layer is (24/2)X(24/2) = 12X12 images
        model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
        # the output of this layer is (12-5+1)X(12-5+1) = 8X8 images 
        model.add(MaxPooling2D(padding="same"))
        # the output of this layer is (8/2)X(8/2) = 4X4
        model.add(Flatten())
        # flatten the 2D input
        #DNN from here on.
        #from the previous Convolution layer we haev 128*4*4 = 2048 numbers 
        #as the input for a single image in the input dataset
        model.add(Dense(2048,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(10,activation="sigmoid"))
        # the output layer
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, batch_size=100, epochs=epochs, validation_data=(self.x_test, self.y_test), verbose = verbose)
        train_results = model.evaluate(self.x_train, self.y_train, verbose=verbose)
        test_results = model.evaluate(self.x_test, self.y_test, verbose=verbose)
        self.results = {'test_results': test_results, 'train_results':train_results}
        return "Model Successfully Compiled", 0
    def check_results(self):
        if(self.results):
            df = pd.DataFrame(self.results, index=['Loss', 'Percent Accuracy'])
            self.df_results = df
            print(df)
        else:
            return "Model Not Compiled", -1


# Checking the results for DNN on MNIST
# temp = mnist_dnn()
# neurons = [100, 200, 200, 200, 100]
# activations = ['relu']*5
# print(temp.dnn_output(5, neurons, activations, 10))
# temp.check_results()

# checking the results for CNn on MNIST
# temp_1 = mnist_cnn()
# print(temp_1.normalize_data())
# temp_1.cnn_output(verbose=1, epochs=10)
# temp_1.check_results()