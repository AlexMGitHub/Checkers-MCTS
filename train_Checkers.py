#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# train_Checkers.py
#
# Revision:     1.00
# Date:         11/27/2020
# Author:       Alex
#
# Purpose:      Trains a neural network to play Checkers using self-play data.  
#               The initial dataset can be generated via randomized rollouts.
#               Once a neural network has been trained on the initial dataset,
#               subsequent datasets can be generated using the neural network's
#               self-play.
#
# Inputs:
# 1. The neural network architecture as defined in the create_nn() function.
# 2. Various parameters for generating training data.
# 3. Various parameters for training the neural network.
# 4. Various parameters for evaluating the NNs in a Checkers tournament.
#
# Outputs:
# 1. Saves the trained neural networks to /data/model.
# 2. Saves plots of the NN training loss to /data/plots.
#
###############################################################################
"""
# %% Imports
from Checkers import Checkers
from training_pipeline import generate_Checkers_data
from training_pipeline import tournament_Checkers
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import Sequence
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf


# %% Functions
def create_nn(LR_INITIAL, conv_reg=0.0001, dense_reg=0.0001):
    """Create a double-headed neural network used to learn to play Checkers."""
    creg = l2(conv_reg) # Conv2D regularization param
    dreg = l2(dense_reg) # Dense regularization param
    inputs = Input(shape = (14,8,8))
    conv0 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(inputs)
    conv1 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(conv0)
    conv2 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(conv1)
    conv3 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(conv2)
    conv4 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(conv3)
    conv5 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(conv4)
    conv6 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(conv5)
    # Create policy head
    policy_conv1 = Conv2D(8, (3, 3), padding='same', activation = 'relu', 
                          use_bias = True, data_format='channels_first',
                          kernel_regularizer=creg, bias_regularizer=creg)(conv6)
    policy_conv2 = Conv2D(4, (1, 1), padding='same', activation = 'relu', 
                          use_bias = True, data_format='channels_first',
                          kernel_regularizer=creg, bias_regularizer=creg)(policy_conv1)
    policy_flat1 = Flatten()(policy_conv2)
    policy_output = Dense(256, activation = 'softmax', use_bias = True,
                          kernel_regularizer=dreg, bias_regularizer=dreg)(policy_flat1)
    # Create value head
    value_conv1 = Conv2D(1, (1, 1), padding='same', activation = 'relu', 
                         use_bias = True, data_format='channels_first',
                         kernel_regularizer=creg, bias_regularizer=creg)(conv6)
    value_flat1 = Flatten()(value_conv1)
    value_dense1 = Dense(64, activation='relu', use_bias = True,
                         kernel_regularizer=dreg, bias_regularizer=dreg)(value_flat1)
    value_output = Dense(1, activation='tanh', use_bias = True,
                         kernel_regularizer=dreg, bias_regularizer=dreg)(value_dense1)
    # Compile model
    model = Model(inputs, [policy_output, value_output])
    model.compile(loss=['kullback_leibler_divergence', 'mse'], 
                  loss_weights = [1.0, 1.0], optimizer=Adam(lr=LR_INITIAL))
    return model

# @tf.autograph.experimental.do_not_convert
# def train_nn(training_data, neural_network, val_split=0):
#     """"""
#     early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
#                                                   patience=3, mode='min', 
#                                                   min_delta=0.1, verbose=1)

#     #mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
#     states = np.array([e[0][:14] for e in training_data])
#     probs = np.array([e[1]for e in training_data])
#     qvals = np.array([e[2]for e in training_data])
#     history = neural_network.fit(states, [probs, qvals], 
#                        verbose=True, epochs=EPOCHS, batch_size=BATCH_SIZE,
#                        validation_split=val_split, shuffle=True,
#                        callbacks=[early_stop])
#     return history

@tf.autograph.experimental.do_not_convert
def train_nn(training_data, neural_network, val_split=0):
    """"""
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=3, mode='min', 
                                                  min_delta=0.01, verbose=1)

    #mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    np.random.shuffle(training_data)
    if val_split > 0:
        validation_data = training_data[-int(len(training_data)*val_split):]
        del training_data[-int(len(training_data)*val_split):]
        validation_generator = My_Custom_Generator(validation_data, BATCH_SIZE)
        validation_steps = int(len(validation_data) // BATCH_SIZE)
    else:
        validation_generator = None
        validation_steps = None
        
    
    training_generator = My_Custom_Generator(training_data, BATCH_SIZE)
    steps_per_epoch = int(len(training_data) // BATCH_SIZE)
    
    # states = np.array([e[0][:14] for e in training_data])
    # probs = np.array([e[1]for e in training_data])
    # qvals = np.array([e[2]for e in training_data])
    # history = neural_network.fit(states, [probs, qvals], 
    #                    verbose=True, epochs=EPOCHS, batch_size=BATCH_SIZE,
    #                    validation_split=val_split, shuffle=True,
    #                    callbacks=[early_stop])
    history = neural_network.fit(x = training_generator,
                                 steps_per_epoch = steps_per_epoch,
                                 validation_data = validation_generator,
                                 validation_steps = validation_steps,
                                 epochs = EPOCHS,
                                 verbose = 1,
                                 shuffle = True,
                                 callbacks=[early_stop])
                                           
    return history

def set_nn_lrate(neural_network, lrate):
    """"""
    K.set_value(neural_network.optimizer.learning_rate, lrate)
    
def set_nn_creg(neural_network, creg):
    """"""
    K.set_value(neural_network.optimizer.learning_rate, creg)

def set_nn_dreg(neural_network, dreg):
    """"""
    pass
    
def save_nn_to_disk(neural_network, iteration, timestamp):
    """Save neural network to disk with timestamp and generation in filename."""
    filename = 'data/model/Checkers_Model' + str(iteration) + '_' + timestamp + '.h5'
    neural_network.save(filename)
    return filename
    
def create_timestamp():
    """Create timestamp string to be used in filenames."""
    timestamp = datetime.now(tz=None)
    timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
    return timestamp_str

def plot_history(history):
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # summarize history for loss
    legend = list(history.history.keys())
    for idx in range(len(legend)):
        legend[idx] = legend[idx].replace('dense_3', 'policy_head')
        legend[idx] = legend[idx].replace('dense_5', 'value_head')
    for key in history.history.keys():
        plt.plot(history.history[key])
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Iteration {} Model Loss'.format(TRAINING_ITERATION))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper right')
    filename = 'data/plots/Checkers_Model' + str(TRAINING_ITERATION+1) + \
        '_TrainingLoss_' + create_timestamp() + '.png'
    plt.draw()
    fig1 = plt.gcf()
    fig1.set_dpi(200)
    fig1.savefig(filename)
    plt.show()
    plt.close()

def load_training_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


# %% Classes
class My_Custom_Generator(Sequence):
  """https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71"""
  def __init__(self, data, batch_size) :
    self.data = training_data
    self.batch_size = batch_size
    
    
  def __len__(self):
    return (np.ceil(len(self.data) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx):
    data = self.data[idx * self.batch_size : (idx+1) * self.batch_size]
    states = np.array([e[0][:14] for e in data])
    probs = np.array([e[1]for e in data])
    qvals = np.array([e[2]for e in data])
    return (states, [probs, qvals])



# %% Training loop
#new_nn_fn = 'model/Checkers_Model1_16-Dec-2020(19:00:37).h5'
TRAINING_ITERATION = 0 # Number of training iterations
TOURNEY_GAMES = 10 # Number of games in tournament between NNs

# #-- Set training parameters
nn_lrate = 0.0001 # Neural network learning rate for training
BATCH_SIZE = 512 # Batch size for training neural network
EPOCHS = 100 # Number of training epochs
TRUNCATE_CNT = 120 # Number of moves before truncating training game

# if TRAINING_ITERATION == 0: # Use random rollouts to generate first dataset
#     NEURAL_NET = False
#     NUM_TRAINING_GAMES = 1000
#     CONSTRAINT = 'rollout' # Constraint can be 'rollout' or 'time'
#     BUDGET = 200 # Maximum number of rollouts or time in seconds
#     MULTIPROC = False # Enable multiprocessing
# else: 
#     NEURAL_NET = True
#     NUM_TRAINING_GAMES = 1000
#     CONSTRAINT = 'rollout' # Constraint can be 'rollout' or 'time'
#     BUDGET = 200 # Maximum number of rollouts or time in seconds
#     MULTIPROC = False # Enable multiprocessing

# if NEURAL_NET:
#     nn = load_model(new_nn_fn)
#     game_env = Checkers(nn) # Instantiate Checkers environment
# else:
#     game_env= Checkers()
# UTC_C = 1/(2**0.5) # Constant C used to calculate UCT value
# VERBOSE = False # MCTS prints search start/stop messages if True
# TRAINING = True # True if training NN, False if competitive play
# DIRICHLET_ALPHA = 3.6 # Used to add noise to prior probabilities of actions
# DIRICHLET_EPSILON = 0.25 # Used to add noise to prior probabilities of actions    
# TEMPERATURE_TAU = 1.0 # Initial value of temperature Tau
# TEMPERATURE_DECAY = 0.1 # Linear decay of Tau per move
# TEMP_DECAY_DELAY = 10 # Move count before beginning decay of Tau value

# mcts_kwargs = {
#     'GAME_ENV' : game_env,
#     'UTC_C' : UTC_C,
#     'CONSTRAINT' : CONSTRAINT,
#     'BUDGET' : BUDGET,
#     'MULTIPROC' : MULTIPROC,
#     'NEURAL_NET' : NEURAL_NET,
#     'VERBOSE' : VERBOSE,
#     'TRAINING' : TRAINING,
#     'DIRICHLET_ALPHA' : DIRICHLET_ALPHA,
#     'DIRICHLET_EPSILON' : DIRICHLET_EPSILON,
#     'TEMPERATURE_TAU' : TEMPERATURE_TAU,
#     'TEMPERATURE_DECAY' : TEMPERATURE_DECAY,
#     'TEMP_DECAY_DELAY' : TEMP_DECAY_DELAY
#     }
    
#-- Generate training data and train neural network
# chk_data = generate_Checkers_data(NUM_TRAINING_GAMES, TRAINING_ITERATION,
#                                   TRUNCATE_CNT, **mcts_kwargs)
# training_data = chk_data.generate_data()
training_data = load_training_data('data/training_data/Checkers_Data0_16-Dec-2020(22:00:07).pkl') # cleaned up data
#training_data = load_training_data('data/training_data/Checkers_Data0_06-Dec-2020(14:46:24).pkl') # Bugged data
training_data.extend(load_training_data('data/training_data/Checkers_Data0_15-Dec-2020(17:49:02).pkl')) # Trunc data


if TRAINING_ITERATION == 0: 
    nn = create_nn(nn_lrate)
    old_nn_fn = save_nn_to_disk(nn, TRAINING_ITERATION, create_timestamp())
else:
    old_nn_fn = new_nn_fn

# Train NN
set_nn_lrate(nn, nn_lrate)
history = train_nn(training_data, nn, val_split=0.10)
plot_history(history)
new_nn_fn = save_nn_to_disk(nn, TRAINING_ITERATION+1, create_timestamp())

# Enter new and old NNs into tournament
tourney_kwargs = { # Parameters for MCTS used in tournament
    'GAME_ENV' : Checkers(nn),
    'UTC_C' : 1/(2**0.5),
    'CONSTRAINT' : 'rollout',
    'BUDGET' : 200,
    'MULTIPROC' : False,
    'NEURAL_NET' : True,
    'VERBOSE' : False,
    'TRAINING' : False,
    'DIRICHLET_ALPHA' : 3.6,
    'DIRICHLET_EPSILON' : 0.25,
    'TEMPERATURE_TAU' : 0,
    'TEMPERATURE_DECAY' : 0,
    'TEMP_DECAY_DELAY' : 0
    }
print('Beginning tournament between {} and {}!'.format(new_nn_fn, old_nn_fn))
tourney = tournament_Checkers(new_nn_fn, old_nn_fn, TOURNEY_GAMES, **tourney_kwargs)
tourney.start_tournament()


