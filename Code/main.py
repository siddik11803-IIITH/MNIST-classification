# importing all the required packages

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


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
        model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
        for i in range((layers)): # now we shall iterate through each of the layers
            # thus keep on builiding the model
            model.add(tf.keras.layers.Dense(neurons[i], activation = activations[i]))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
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
            print("Evaluate A model first")
    
temp = mnist_dnn()
neurons = [100, 200, 200, 200, 100]
activations = ['relu']*5
print(temp.dnn_output(5, neurons, activations, 10))
temp.check_results()
    