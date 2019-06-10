import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from neuralNet import NeuralNet



def main():

    np.random.seed(1)



    data_path = 'data/PSbinned.txt'
    data = pd.read_csv(data_path, delim_whitespace = True)

    data = data.drop(['p4', 'p5', 'p7','p8'], axis = 1)

    #split data into training/testing sets, and input/output sets
    training, testing = train_test_split(data, test_size = 0.2)
    train_x, test_x = training.filter(regex = 'z'), testing.filter(regex = 'z')
    train_y, test_y = training.filter(regex = 'p'), testing.filter(regex = 'p')


    #Transform the data logarithmically - easier for the network to learn
    train_y, test_y = np.log10(train_y), np.log10(test_y)

    #Normalize the data to [0,1]
    scale_x = preprocessing.MinMaxScaler().fit(train_x)
    scale_y = preprocessing.MinMaxScaler().fit(train_y)
    train_x, test_x = scale_x.transform(train_x), scale_x.transform(test_x)
    train_y, test_y = scale_y.transform(train_y), scale_y.transform(test_y)

    #Set the hyperparameters
    network = NeuralNet(INPUT_DIM     = len(train_x[0]), OUTPUT_DIM   = len(train_y[0]),
                        HIDDEN_DIM_1  = 150,             HIDDEN_DIM_2 = 100,
                        BATCH_SIZE    = 64,              LOSS         = 'mse',
                        LEARNING_RATE = 0.01,            EPOCHS       = 2000,
                        DROPOUT       = 0.2
                        )
    #Builds/compiles the network
    network.compile()

    #functions applied at given stages of network training
    network.add_callbacks(reduce_lr = 1, early_stop = 1)

    #Function to train the network, Test set used for validation
    network.fit(train_x, train_y, test_x, test_y)

    #Uses trained network to predict outputs/loss for validation set
    network.predict(test_x ,test_y)



    network.predictions      = 10**(scale_y.inverse_transform(network.predictions))
    network.expected_outputs = 10**(scale_y.inverse_transform(network.expected_outputs))


    print('Testing Loss =  ', network.accuracy[0])


    network.plot_predictions(save_fig = 'no')
    network.plot_loss(save_fig = 'no')
    plt.show()








if __name__ == '__main__':
    main()
