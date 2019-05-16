from neuralNet import NeuralNet
import numpy as np
import pandas as pd
import scipy as scp
import matplotlib.pyplot as plt
import os

from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime

from keras import models
from keras import layers
from keras import optimizers
from keras import initializers
from keras import callbacks
from keras import regularizers



def main():


    np.random.seed(100)

    path = '../data/PSbinned_13.1.txt'
    data = pd.read_csv(path)

    #data['avg'] = data.filter(regex = 'z').mean(axis = 1)
    data = data.drop(['p4','p5','p7','p8'], axis = 1)

    training, testing = train_test_split(data, test_size = 0.20)


    train_x = training.filter(regex = 'z')
    test_x  =  testing.filter(regex = 'z')
    train_y = training.filter(regex = 'p')
    test_y  =  testing.filter(regex = 'p')

    train_y, test_y = np.log10(train_y), np.log10(test_y)
    train_x, test_x = np.array(train_x.values.tolist()), np.array(test_x.values.tolist())


    scale_x = preprocessing.MinMaxScaler().fit(train_x)
    scale_y = preprocessing.MinMaxScaler().fit(train_y)
    train_x, test_x = scale_x.transform(train_x), scale_x.transform(test_x)
    train_y, test_y = scale_y.transform(train_y), scale_y.transform(test_y)



    net = NeuralNet(INPUT_DIM = len(train_x[0]), OUTPUT_DIM = len(train_y[0]),
                    HIDDEN_DIM_1 = 150, HIDDEN_DIM_2 = 100, EPOCHS = 5000, BATCH_SIZE = 250,
                    LOSS = 'mse')

    net.compile()
    net.add_callbacks(reduce_lr = 1, early_stop = 1, tensorboard = 0, path = '../outputs/')
    net.fit(train_x, train_y, test_x, test_y)
    net.predict(test_x, test_y)



    net.predictions, net.y_values = scale_y.inverse_transform(net.predictions), scale_y.inverse_transform(net.y_values)
    net.predictions, net.y_values = 10**(net.predictions), 10**(net.y_values)
    net.data_used = ' '

    net.save_outputs(path = '../outputs/')
    net.plot_predictions(save_fig = 'yes', path = '../outputs/')
    net.plot_loss(save_fig = 'yes', path = '../outputs/')


if __name__ == '__main__':
    main()
