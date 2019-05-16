import numpy as np
import pandas as pd
import scipy as scp
import os
import matplotlib.pyplot as plt

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


class NeuralNet :

    def __init__(self, INPUT_DIM    = None, OUTPUT_DIM   = None, LOSS         = 'mse',
                       HIDDEN_DIM_2 = None, HIDDEN_DIM_3 = None, HIDDEN_DIM_1 = None,
                       BATCH_SIZE   = None, EPOCHS       = 1000, DROPOUT      = 0.2,
                       ACTIVATION   = layers.ELU(alpha = 0.1),  LEARNING_RATE = 0.01,
                       METRICS      = ['mae', 'mse']
                       ):


        self.input_dim      = INPUT_DIM
        self.output_dim     = OUTPUT_DIM
        self.hidden_dim_1   = HIDDEN_DIM_1
        self.hidden_dim_2   = HIDDEN_DIM_2
        self.hidden_dim_3   = HIDDEN_DIM_3

        self.loss           = LOSS
        self.batch_size     = BATCH_SIZE
        self.epochs         = EPOCHS
        self.dropout        = DROPOUT
        self.activation     = ACTIVATION
        self.lr             = LEARNING_RATE
        self.optimizer      = optimizers.adam(lr = self.lr)
        self.metrics        = METRICS
        self.model_datetime = datetime.now().strftime('%d-%mT%H-%M-%S')
        self.data_used      = ''



    def compile(self):
        """
        Builds/compiles model architecture for training.

        Minimum 1 hidden layer, maximum 3

        Model parameters set in class initialisation

        """
        self.model = models.Sequential()

        self.model.add(layers.Dense(
                  self.hidden_dim_1, input_dim = self.input_dim,
                  kernel_initializer = initializers.RandomNormal(seed = 1),
                  bias_initializer   = initializers.Constant(0.01),
                  kernel_regularizer=regularizers.l2(0.0005)))

        self.model.add(self.activation)
        self.model.add(layers.Dropout(self.dropout))

        if self.hidden_dim_2 != None:
            self.model.add(layers.Dense(self.hidden_dim_2, kernel_regularizer=regularizers.l2(0.0005)))
            self.model.add(self.activation)
            self.model.add(layers.Dropout(self.dropout))


        if self.hidden_dim_3 != None:
            self.model.add(layers.Dense(self.hidden_dim_3))
            self.model.add(self.activation)
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Dense(self.output_dim))
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)


        self.weights = self.model.get_weights()

        print('Model Created')
        return self.model






    def add_callbacks(self, reduce_lr = 0, early_stop = 0, tensorboard = 0, path = None):
        """
        Adds callbacks to keras model. Set value to 1 to activate

          reduce_lr : reduces learning rate by a factor of 10 for plateau in
                      validation loss with patience of 100 epochs

         early_stop : stops model training for plateau in validation loss with
                      patience of 200 epochs

        tensorboard : writes a log to subdirectory path +/logs for visualisation of
                      model learning. More info :
                      https://www.tensorflow.org/guide/summaries_and_tensorboard

               path : directory to write tensorboard logs

        """
        self.callbacks_list = []

        if reduce_lr == 1:
            reduce_lr = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                    factor = 0.3, patience = 50,
                                                    min_lr = 10**-4)
            self.callbacks_list.append(reduce_lr)

        if early_stop == 1:
            early_stop = callbacks.EarlyStopping (monitor = 'val_loss',
                                                  patience = 100,
                                                  restore_best_weights = True)
            self.callbacks_list.append(early_stop)


        if tensorboard == 1:

            if path != None:
                path = path +'logs'
                if not os.path.exists(path):
                    os.makedirs(path)
            else:
                path = 'logs'
                if not os.path.exists(path):
                    os.makedirs(path)

            tensorboard = callbacks.TensorBoard(log_dir =path +"/"+self.model_datetime,
                                                batch_size = 100, histogram_freq = 100,
                                                update_freq = 'epoch')
            self.callbacks_list.append(tensorboard)

        return self.callbacks_list





    def fit(self, x_training, y_training, x_testing, y_testing):
        """
        Trains the model for training set x_training, y_training
        and validates on x_testing, y_testing

        Updates keras model and history objects

        """
        start = time()
        self.history = self.model.fit(x_training, y_training,
                            epochs = self.epochs , batch_size = self.batch_size,
                            verbose = 2, validation_data = (x_testing, y_testing),
                            callbacks = self.callbacks_list, shuffle = True
                            )
        end = time()

        self.time_to_fit = end - start

        return





    def predict(self, x_testing, y_testing):
        """
        Returns trained model predicted outputs for given input array



        """
        self.y_values    = y_testing
        self.accuracy    = self.model.evaluate(x_testing,y_testing)
        self.predictions = self.model.predict(x_testing)


        return





    def save_outputs(self, path = None) :
        """
        Function to save outputs produced by fit & predict functions

        saves to folder with path path + /datetime

        """

        if path != None:
            path = path + self.model_datetime
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = self.model_datetime
            if not os.path.exists(path):
                os.makedirs(path)

        np.savetxt(path + '/y_values.txt',   self.y_values , delimiter = " ")
        np.savetxt(path + '/predictions.txt',self.predictions , delimiter = " ")
        np.savetxt(path + '/loss.txt',       self.history.history['loss'] , delimiter = " ")
        np.savetxt(path + '/val_loss.txt',   self.history.history['val_loss'] , delimiter = " ")
        np.savetxt(path + '/mae.txt',        self.history.history['mean_absolute_error'] , delimiter = " ")
        np.savetxt(path + '/val_mae.txt',    self.history.history['val_mean_absolute_error'] , delimiter = " ")
        np.savetxt(path + '/lr.txt',         self.history.history['lr'] , delimiter = " ")


        parameter_file = path+'/model_description.txt'
        file = open(parameter_file, 'w')

        file.write('   Data Used  : ' + str(self.data_used)    + '\n')
        file.write('   INPUT_DIM  : ' + str(self.input_dim)    + '\n')
        file.write('  HIDDEN_DIM  : ' + str(self.hidden_dim_1) + '\n')
        file.write('HIDDEN_DIM_2  : ' + str(self.hidden_dim_2) + '\n')
        file.write('HIDDEN_DIM_3  : ' + str(self.hidden_dim_3) + '\n')
        file.write('     DROPOUT  : ' + str(self.dropout)      + '\n')
        file.write('  BATCH_SIZE  : ' + str(self.batch_size)   + '\n')
        file.write('      EPOCHS  : ' + str(len(self.history.history['loss']))+ '\n')
        file.write('  TIME_TO_FIT : ' + str(self.time_to_fit)  + '\n')
        file.write('      VAL_MSE : ' + str(self.accuracy[2])  + '\n')
        file.write('      VAL_MAE : ' +  str(self.accuracy[1]) + '\n')
        file.close()

        return





    def plot_predictions(self, save_fig = 'no', path = ''):
        """
        Produces/saves .png of expected/predicted parameters

        """
        axis_x = [r'$\mathit{f}_{\ast,true}$', r'$V_{circ,true}$',r'$\mathit{f}_{X,true}$', r'$\tau_{CMB,true}$' ]
        axis_y = [r'$\mathit{f}_{\ast,pred}$', r'$V_{circ,pred}$',r'$\mathit{f}_{X,pred}$', r'$\tau_{CMB,pred}$' ]

        fig = plt.figure(figsize = (800/50, 800/96), dpi = 96)

        for i in range(len(self.y_values[0])):

            x = list(map(list, zip(*self.y_values)))[i]
            y = list(map(list, zip(*self.predictions)))[i]

            ax = plt.subplot(2,2,i+1)

            if (i == 1) or (i==0) or (i ==2) or (i ==3):
                ax.loglog( x, y, '+')

                ax.set_xlim(min(min(x) - 0.1*min(x), min(y) - 0.1*min(y)),
                            max(max(x) + 0.4*max(x), max(y) + 0.4*max(y)))
                ax.set_ylim(min(min(x) - 0.1*min(x), min(y) - 0.1*min(y)),
                            max(max(x) + 0.4*max(x), max(y) + 0.4*max(y)))

                x_vals = np.array(plt.xlim())
                y_vals = x_vals
                ax.loglog(x_vals, y_vals, '--', color = 'k', linewidth = 1.0)

            else:
                ax.plot( x, y, '+')
                ax.set_xlim(min(min(x) - 0.1*min(x), min(y) - 0.1*min(y)),
                            max(max(x) + 0.1*max(x), max(y) + 0.1*max(y)))
                ax.set_ylim(min(min(x) - 0.1*min(x), min(y) - 0.1*min(y)),
                            max(max(x) + 0.1*max(x), max(y) + 0.1*max(y)))
                x_vals = np.array(plt.xlim())
                y_vals = x_vals
                ax.plot(x_vals, y_vals, '--', color = 'k', linewidth = 1.0)

            ax.grid()
            ax.set_xlabel(axis_x[i], size = 13)
            ax.set_ylabel(axis_y[i], size = 13)

        fig.subplots_adjust(top = 0.9, bottom = 0.1, hspace = 0.3, wspace = 0.25, right = 0.93, left = 0.08)
        #plt.show()

        if (save_fig == 'yes'):
            fig.savefig(path + self.model_datetime +'/predictions.png', dpi = 100)

        return






    def plot_loss(self, save_fig = 'no', path = ''):
        """
        Plots loss metrics - Mean Square Error & Mean Absolute Error


        """

        fig1 = plt.figure(figsize = (800/50, 800/96), dpi = 96)
        fig1.subplots_adjust(top = 0.9, bottom = 0.1, hspace = 0.3, wspace = 0.25, right = 0.93, left = 0.08)

        ax1 = plt.subplot(1,2,1)

        ax1.set_ylabel('MSE')
        ax1.set_xlabel('Epoch')

        ax1.plot(self.history.history['mean_squared_error'], label = 'Training Loss')
        ax1.plot(self.history.history['val_mean_squared_error'], label = 'Validation Loss')
        ax1.set_xlim([0,len(self.history.history['mean_squared_error'])-1])

        ax3 = ax1.twinx()
        ax3.semilogy(self.history.history['lr'], color = 'k', alpha = 0.4)
        ax3.set_ylabel('Learning Rate')



        ax1.legend()


        ax2 = plt.subplot(1,2,2)

        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.plot(self.history.history['mean_absolute_error'], label = 'Training MAE')
        ax2.plot(self.history.history['val_mean_absolute_error'], label = 'Validation MAE')
        ax2.set_xlim([0,len(self.history.history['mean_absolute_error'])-1])

        ax4 = ax2.twinx()
        ax4.semilogy(self.history.history['lr'], color = 'k' , alpha = 0.4)
        ax4.set_ylabel('Learning Rate')
        ax2.legend()

        if (save_fig == 'yes'):
            fig1.savefig(path + self.model_datetime +'/loss.png', dpi = 100)

        return



    def save_best_network(self):

        return
