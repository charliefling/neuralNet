import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers, activations, regularizers, optimizers

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from neuralNet import NeuralNet
import os
from keras import backend as K
import tensorflow as tf


def main():

    np.random.seed(0)

    data_path = 'data/PSbinned.txt'
    data = pd.read_csv(data_path, delim_whitespace = True)

    #split data into training/testing sets, and input/output sets
    training, testing = train_test_split(data, test_size = 0.10)


    train_y, test_y = training.filter(regex = 'z8'), testing.filter(regex = 'z8')
    train_x, test_x = training.filter(regex = 'p'), testing.filter(regex = 'p')


    #Transform the data logarithmically
    train_x, test_x = np.log10(train_x + 1), np.log10(test_x + 1)
    train_y, test_y = np.log10(train_y + 1), np.log10(test_y+ 1)




    #Normalize the data to [0,1]
    scale_x = preprocessing.MinMaxScaler().fit(train_x)
    scale_y = preprocessing.MinMaxScaler().fit(train_y)
    train_x, test_x = scale_x.transform(train_x), scale_x.transform(test_x)
    train_y, test_y = scale_y.transform(train_y), scale_y.transform(test_y)



    #plt.hist(train_y.T[0], bins = 100)
    #plt.show()


    #Set the hyperparameters
    network = NeuralNet(INPUT_DIM     = len(train_x[0]),   OUTPUT_DIM   = len(train_y[0]),
                        HIDDEN_DIM_1  = 300,               HIDDEN_DIM_2 =  300,
                        BATCH_SIZE    = 128,                LOSS        = 'mse',
                        LEARNING_RATE = 0.01,              EPOCHS       =  5000,
                        DROPOUT       = 0.0,               ACTIVATION   = layers.ELU(alpha = 0.15),
                        REGULARIZER   = None,              OPTIMIZER    = 'adam'
                        )
    #Builds/compiles the network
    network.compile()

    #functions applied at given stages of network training
    network.add_callbacks(reduce_lr = 1, early_stop = 1)

    #Function to train the network, Test set used for validation
    network.fit(train_x, train_y, test_x, test_y)

    #Uses trained network to predict outputs/loss for validation set
    network.predict(test_x ,test_y)



    network.predictions      =  10**(scale_y.inverse_transform(network.predictions)) - 1.0
    network.expected_outputs =  10**( scale_y.inverse_transform(network.expected_outputs)) - 1.0


    print('Testing Loss =  ', network.accuracy[0])

    #uncomment to save network outputs to file
    #network.save_outputs(path = 'outputs/')
    #network.save_best_network(path = 'outputs/')


if __name__ == '__main__':
    main()
