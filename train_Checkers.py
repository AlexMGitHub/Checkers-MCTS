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
from keras.layers import BatchNormalization
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
def create_nn(LR_INITIAL, conv_reg=0.001, dense_reg=0.001):
    """Create a double-headed neural network used to learn to play Checkers."""
    creg = l2(conv_reg) # Conv2D regularization param
    dreg = l2(dense_reg) # Dense regularization param
    inputs = Input(shape = (14,8,8))
    conv0 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(inputs)
    bn0 = BatchNormalization()(conv0)
    conv1 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn0)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn1)
    bn2 = BatchNormalization()(conv2)
    conv3 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn2)
    bn3 = BatchNormalization()(conv3)
    conv4 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn3)
    bn4 = BatchNormalization()(conv4)
    conv5 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn4)
    bn5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_first',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn5)
    bn6 = BatchNormalization()(conv6)
    # Create policy head
    policy_conv1 = Conv2D(8, (3, 3), padding='same', activation = 'relu', 
                      use_bias = True, data_format='channels_first',
                      kernel_regularizer=creg, bias_regularizer=creg)(bn6)
    bn_pol1 = BatchNormalization()(policy_conv1)
    policy_conv2 = Conv2D(4, (1, 1), padding='same', activation = 'relu', 
                      use_bias = True, data_format='channels_first',
                      kernel_regularizer=creg, bias_regularizer=creg)(bn_pol1)
    bn_pol2 = BatchNormalization()(policy_conv2)
    policy_flat1 = Flatten()(bn_pol2)
    policy_output = Dense(256, activation = 'softmax', use_bias = True,
                  kernel_regularizer=dreg, bias_regularizer=dreg,
                  name='policy_head')(policy_flat1)
    # Create value head
    value_conv1 = Conv2D(1, (1, 1), padding='same', activation = 'relu', 
                         use_bias = True, data_format='channels_first',
                         kernel_regularizer=creg, bias_regularizer=creg)(bn6)
    bn_val1 = BatchNormalization()(value_conv1)
    value_flat1 = Flatten()(bn_val1)
    value_dense1 = Dense(64, activation='relu', use_bias = True,
                 kernel_regularizer=dreg, bias_regularizer=dreg)(value_flat1)
    bn_val2 = BatchNormalization()(value_dense1)
    value_output = Dense(1, activation='tanh', use_bias = True,
                     kernel_regularizer=dreg, bias_regularizer=dreg,
                     name='value_head')(bn_val2)
    # Compile model
    model = Model(inputs, [policy_output, value_output])
    model.compile(loss={'policy_head' : 'kullback_leibler_divergence', 
                        'value_head' : 'mse'}, 
                  loss_weights={'policy_head' : POLICY_LOSS_WEIGHT, 
                                  'value_head' : VALUE_LOSS_WEIGHT}, 
                  optimizer=Adam(lr=LR_INITIAL))
    return model

@tf.autograph.experimental.do_not_convert
def train_nn(training_data, neural_network, val_split=0):
    """Trains neural network according to desired parameters."""
    # Create early stop callback for training
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience=PATIENCE, mode='min', 
                                              min_delta=MIN_DELTA, verbose=1)
    np.random.shuffle(training_data) # Randomize order of training data
    if val_split > 0: # Split data into training and validation sets
        validation_data = training_data[-int(len(training_data)*val_split):]
        del training_data[-int(len(training_data)*val_split):]
        validation_generator = Keras_Generator(validation_data, BATCH_SIZE)
        validation_steps = len(validation_generator)
    else:
        validation_generator = None
        validation_steps = None
    # Create generator to feed training data to NN        
    training_generator = Keras_Generator(training_data, BATCH_SIZE)
    steps_per_epoch = len(training_generator)
    # Train NN using generators and early stopping callback
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
    """Set the learning rate of an existing neural network."""
    K.set_value(neural_network.optimizer.learning_rate, lrate)
    
def save_nn_to_disk(neural_network, iteration, timestamp):
    """Save neural network to disk with timestamp and iteration in filename."""
    filename = 'data/model/Checkers_Model' + str(iteration) + '_' + \
        timestamp + '.h5'
    neural_network.save(filename)
    return filename
    
def create_timestamp():
    """Create timestamp string to be used in filenames."""
    timestamp = datetime.now(tz=None)
    timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
    return timestamp_str

def plot_history(history, nn):
    """Plot of training loss versus training epoch and save to disk."""
    legend = list(history.history.keys())
    for key in history.history.keys():
        plt.plot(history.history[key])
    plt.title('Iteration {} Model Loss'.format(TRAINING_ITERATION))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper right')
    plt.grid()
    filename = 'data/plots/Checkers_Model' + str(TRAINING_ITERATION+1) + \
        '_TrainingLoss_' + create_timestamp() + '.png'
    plt.draw()
    fig1 = plt.gcf()
    fig1.set_dpi(200)
    fig1.savefig(filename)
    plt.show()
    plt.close()
    return filename

def load_training_data(filename):
    """Load Pickle file and return training data."""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def record_training_params():
    """Document the parameters used in the training pipeline."""
    filename = 'data/training_params/Checkers_Training_Params' + \
        create_timestamp() + '.txt'
    nn_var_list = ['NN_LRATE', 'BATCH_SIZE', 'EPOCHS', 
                'CONV_REG', 'DENSE_REG', 'VAL_SPLIT', 'MIN_DELTA', 'PATIENCE',
                'POLICY_LOSS_WEIGHT', 'VALUE_LOSS_WEIGHT', 'old_nn_fn', 
                'new_nn_fn']
    with open(filename, 'w') as file:
        file.write('TRAINING_ITERATION = ' + str(TRAINING_ITERATION) + '\n\n')
        file.write('Data Generation Parameters:\n')
        file.write('data_fn = ' + data_fn + '\n')
        file.write('TRUNCATE_CNT = ' + str(TRUNCATE_CNT) + '\n')
        file.write('mcts_kwargs = \n')
        file.write(str(eval('mcts_kwargs')) + '\n\n')
        file.write('Neural Network Parameters:\n')
        for var in nn_var_list:
            file.write(var + ' = ' + str(eval(var)) + '\n')
        file.write('Plot filename = ' + plot_filename + '\n')
        file.write('\nTournament Parameters:\n')
        file.write('tourney_fn = ' + tourney_fn + '\n')
        file.write('TOURNEY_GAMES = ' + str(eval('TOURNEY_GAMES')) + '\n')
        file.write('tourney_kwargs = \n')
        file.write(str(eval('tourney_kwargs')) + '\n\n')
        
        
# %% Classes
class Keras_Generator(Sequence):
    """Generator to feed training/validation data to Keras fit() function."""
    def __init__(self, data, batch_size) :
        self.data = data
        self.batch_size = batch_size
    
    def __len__(self):
        return (np.ceil(len(self.data) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx):
        """Splits data into features and labels.  Returns average of Q-values
        and Z-values for the value head labels.
        """
        data = self.data[idx * self.batch_size : (idx+1) * self.batch_size]
        states = np.array([e[0][:14] for e in data])
        probs = np.array([e[1]for e in data])
        qvals = np.array([e[2]for e in data])
        zvals = np.array([e[3]for e in data])
        return (states, [probs, (qvals+zvals)/2])


# %% Training loop
TRAINING_ITERATION = 0 # Current training iteration
if TRAINING_ITERATION != 0: # Load previous iteration's trained NN
    new_nn_fn = 'data/model/Checkers_Model1_27-Dec-2020(21:57:18).h5'

# Set data generation parameters
TRUNCATE_CNT = 120 # Number of moves before truncating training game
# Set training parameters
NN_LRATE = 0.0001 # Neural network learning rate for training
BATCH_SIZE = 32 # Batch size for training neural network
EPOCHS = 100 # Maximum number of training epochs
CONV_REG = 0.001 # L2 regularization term for Conv2D layers
DENSE_REG = 0.001 # L2 regularization term for Dense layers
VAL_SPLIT = 0.10 # Fraction of training data to use for validation
MIN_DELTA = 0.01 # Min amount validation loss must decrease to prevent stopping
PATIENCE = 3 # Number of epochs of stagnation before stopping training
POLICY_LOSS_WEIGHT = 1.0 # Weighting given to policy head loss
VALUE_LOSS_WEIGHT = 0.25 # Weighting given to value head loss
# Tournament Parameters
TOURNEY_GAMES = 10 # Number of games in tournament between NNs


# %% Set MCTS Parameters
if TRAINING_ITERATION == 0: # Use random rollouts to generate first dataset
    NEURAL_NET = False
    NUM_TRAINING_GAMES = 1
    CONSTRAINT = 'rollout' # Constraint can be 'rollout' or 'time'
    BUDGET = 200 # Maximum number of rollouts or time in seconds
    MULTIPROC = False # Enable multiprocessing
else: 
    NEURAL_NET = True
    NUM_TRAINING_GAMES = 1000
    CONSTRAINT = 'rollout' # Constraint can be 'rollout' or 'time'
    BUDGET = 200 # Maximum number of rollouts or time in seconds
    MULTIPROC = False # Enable multiprocessing

if NEURAL_NET:
    nn = load_model(new_nn_fn)
    game_env = Checkers(nn) # Instantiate Checkers environment with NN
else:
    game_env= Checkers()
UCT_C = 4#1/(2**0.5) # Constant C used to calculate UCT value
VERBOSE = False # MCTS prints search start/stop messages if True
TRAINING = True # True if training NN, False if competitive play
DIRICHLET_ALPHA = 1.0#3.6 # Used to add noise to prior probabilities of actions
DIRICHLET_EPSILON = 0.25 # Used to add noise to prior probabilities of actions    
TEMPERATURE_TAU = 1.0 # Initial value of temperature Tau
TEMPERATURE_DECAY = 0.1 # Linear decay of Tau per move
TEMP_DECAY_DELAY = 10 # Move count before beginning decay of Tau value

mcts_kwargs = { # Parameters for MCTS used in generating data
    'GAME_ENV' : game_env,
    'UCT_C' : UCT_C,
    'CONSTRAINT' : CONSTRAINT,
    'BUDGET' : BUDGET,
    'MULTIPROC' : MULTIPROC,
    'NEURAL_NET' : NEURAL_NET,
    'VERBOSE' : VERBOSE,
    'TRAINING' : TRAINING,
    'DIRICHLET_ALPHA' : DIRICHLET_ALPHA,
    'DIRICHLET_EPSILON' : DIRICHLET_EPSILON,
    'TEMPERATURE_TAU' : TEMPERATURE_TAU,
    'TEMPERATURE_DECAY' : TEMPERATURE_DECAY,
    'TEMP_DECAY_DELAY' : TEMP_DECAY_DELAY
    }

if TRAINING_ITERATION == 0: 
    nn = create_nn(NN_LRATE, conv_reg=CONV_REG, dense_reg=DENSE_REG)
    old_nn_fn = save_nn_to_disk(nn, TRAINING_ITERATION, create_timestamp())
else:
    old_nn_fn = new_nn_fn

tourney_kwargs = { # Parameters for MCTS used in tournament
    'GAME_ENV' : Checkers(nn),
    'UCT_C' : UCT_C,
    'CONSTRAINT' : CONSTRAINT,
    'BUDGET' : BUDGET,
    'MULTIPROC' : False,
    'NEURAL_NET' : True,
    'VERBOSE' : False,
    'TRAINING' : False,
    'DIRICHLET_ALPHA' : DIRICHLET_ALPHA,
    'DIRICHLET_EPSILON' : DIRICHLET_EPSILON,
    'TEMPERATURE_TAU' : 0,
    'TEMPERATURE_DECAY' : 0,
    'TEMP_DECAY_DELAY' : 0
    }
    

# %% Generate training data and train neural network
chk_data = generate_Checkers_data(NUM_TRAINING_GAMES, TRAINING_ITERATION,
                                  TRUNCATE_CNT, **mcts_kwargs)
training_data, data_fn = chk_data.generate_data()

# Load existing training data (comment out if generating new data)
# data_fn = 'data/training_data/Checkers_Data0_24-Dec-2020(19:57:37).pkl' # Rewards flipped
# training_data = load_training_data(data_fn) 


# # %% Train NN
# set_nn_lrate(nn, NN_LRATE)
# history = train_nn(training_data, nn, val_split=VAL_SPLIT)
# plot_filename = plot_history(history, nn)
# new_nn_fn = save_nn_to_disk(nn, TRAINING_ITERATION+1, create_timestamp())


# # %% Enter new and old NNs into tournament
# print('Beginning tournament between {} and {}!'.format(new_nn_fn, old_nn_fn))
# tourney = tournament_Checkers(new_nn_fn, old_nn_fn, TOURNEY_GAMES, 
#                               **tourney_kwargs)
# tourney_fn = tourney.start_tournament()
# record_training_params()