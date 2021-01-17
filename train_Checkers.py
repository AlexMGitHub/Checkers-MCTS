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
# Uncomment following 3 lines to force TF to use the CPU instead of GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from keras.layers import Reshape
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import Sequence
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from CLR.clr_callback import CyclicLR
from LRFinder.keras_callback import LRFinder


# %% Functions
def create_nn(conv_reg, dense_reg):
    """Create a double-headed neural network used to learn to play Checkers."""
    creg = l2(conv_reg) # Conv2D regularization param
    dreg = l2(dense_reg) # Dense regularization param
    num_kernels = 128 # Number of Conv2D kernels in "body" of NN
    inputs = Input(shape = (8,8,14))
    conv0 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(inputs)
    bn0 = BatchNormalization(axis=-1)(conv0)
    conv1 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn0)
    bn1 = BatchNormalization(axis=-1)(conv1)
    conv2 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn1)
    bn2 = BatchNormalization(axis=-1)(conv2)
    conv3 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn2)
    bn3 = BatchNormalization(axis=-1)(conv3)
    conv4 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn3)
    bn4 = BatchNormalization(axis=-1)(conv4)
    conv5 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn4)
    bn5 = BatchNormalization(axis=-1)(conv5)
    conv6 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                   use_bias = True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn5)
    bn6 = BatchNormalization(axis=-1)(conv6)
    # Create policy head
    policy_conv1 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                      use_bias = True, data_format='channels_last',
                      kernel_regularizer=creg, bias_regularizer=creg)(bn6)
    bn_pol1 = BatchNormalization(axis=-1)(policy_conv1)
    policy_conv2 = Conv2D(8, (1, 1), padding='same', activation = 'relu', 
                      use_bias = True, data_format='channels_last',
                      kernel_regularizer=creg, bias_regularizer=creg)(bn_pol1)
    bn_pol2 = BatchNormalization(axis=-1)(policy_conv2)
    policy_flat1 = Flatten()(bn_pol2)
    policy_output = Dense(512, activation = 'softmax', use_bias = True,
                  kernel_regularizer=dreg, bias_regularizer=dreg,
                  name='policy_head')(policy_flat1)
    # Create value head
    value_conv1 = Conv2D(1, (1, 1), padding='same', activation = 'relu', 
                         use_bias = True, data_format='channels_last',
                         kernel_regularizer=creg, bias_regularizer=creg)(bn6)
    bn_val1 = BatchNormalization(axis=-1)(value_conv1)
    value_flat1 = Flatten()(bn_val1)
    value_dense1 = Dense(64, activation='relu', use_bias = True,
                 kernel_regularizer=dreg, bias_regularizer=dreg)(value_flat1)
    bn_val2 = BatchNormalization(axis=-1)(value_dense1)
    value_output = Dense(1, activation='tanh', use_bias = True,
                     kernel_regularizer=dreg, bias_regularizer=dreg,
                     name='value_head')(bn_val2)
    # Compile model
    model = Model(inputs, [policy_output, value_output])
    model.compile(loss={'policy_head' : 'categorical_crossentropy',
                        'value_head' : 'mse'}, 
                  loss_weights={'policy_head' : POLICY_LOSS_WEIGHT, 
                                  'value_head' : VALUE_LOSS_WEIGHT}, 
                  optimizer=Adam())
    return model

#@tf.autograph.experimental.do_not_convert
def train_nn(training_data, neural_network, val_split=0):
    """Trains neural network according to desired parameters."""
    # Create early stop callback for training
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience=PATIENCE, mode='min', 
                                              min_delta=MIN_DELTA, verbose=1)
    # Create model checkpoint callback to save best model
    filepath = 'data/model/Checkers_Model' + str(TRAINING_ITERATION+1) + '_' \
        + create_timestamp() + '.h5'
    save_best = tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                   monitor='val_loss', 
                                                   verbose=1, 
                                                   save_best_only=True,
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   save_freq='epoch')
    # Create data generators for training
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
    # Set CLR options
    clr_step_size = int(CLR_SS_COEFF * (len(training_data)/BATCH_SIZE))
    base_lr = NN_BASE_LR
    max_lr = NN_MAX_LR
    mode='triangular'
    # Define the CLR callback
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=mode)
    # Train NN using generators and callbacks
    history = neural_network.fit(x = training_generator,
                                 steps_per_epoch = steps_per_epoch,
                                 validation_data = validation_generator,
                                 validation_steps = validation_steps,
                                 epochs = EPOCHS,
                                 verbose = 1,
                                 shuffle = True,
                                 callbacks=[early_stop, clr, save_best])
    return history, filepath

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
    nn_var_list = ['NN_BASE_LR', 'NN_MAX_LR', 'CLR_SS_COEFF', 'BATCH_SIZE', 'EPOCHS', 
                'CONV_REG', 'DENSE_REG', 'VAL_SPLIT', 'MIN_DELTA', 'PATIENCE',
                'POLICY_LOSS_WEIGHT', 'VALUE_LOSS_WEIGHT', 'old_nn_fn', 
                'new_nn_fn']
    with open(filename, 'w') as file:
        file.write('TRAINING_ITERATION = ' + str(TRAINING_ITERATION) + '\n\n')
        file.write('Data Generation Parameters:\n')
        file.write('data_fn = ' + data_fn + '\n')
        file.write('TERMINATE_CNT = ' + str(TERMINATE_CNT) + '\n')
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
        
def run_lr_finder(training_data, start_lr=1e-8, end_lr=1e-1, num_epochs=6):
    """Linearly increase learning rate while training neural network over
    a number of epochs.  Outputs plot of training loss versus training
    iteration used to select the base and maximum learning rates used in CLR.
    """
    np.random.shuffle(training_data) # Randomize order of training data
    model = create_nn(conv_reg=CONV_REG, dense_reg=DENSE_REG)
    # Define LR finder callback
    lr_finder = LRFinder(min_lr=start_lr, max_lr=end_lr)
    # Create generator to feed training data to NN   
    training_generator = Keras_Generator(training_data, BATCH_SIZE)
    steps_per_epoch = len(training_generator)
    # Perform LR finder
    model.fit(x = training_generator,
                steps_per_epoch = steps_per_epoch,
                validation_data = None,
                validation_steps = None,
                epochs = num_epochs,
                verbose = 1,
                shuffle = True,
                callbacks=[lr_finder])
        
    
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
        states = np.moveaxis(states,1,-1) # Channels last format
        probs = np.array([np.array(e[1]).flatten() for e in data])
        qvals = np.array([e[2]for e in data])
        zvals = np.array([e[3]for e in data])
        return (states, [probs, (qvals+zvals)/2])


# %% Set training pipeline parameters
TRAINING_ITERATION = 0 # Current training iteration
if TRAINING_ITERATION != 0: # Load previous iteration's trained NN
    new_nn_fn = 'data/model/Checkers_Model0_16-Jan-2021(00:12:16).h5'

# Set data generation parameters
TERMINATE_CNT = 160 # Number of moves before terminating training game
# Set training parameters
NN_BASE_LR = 5e-5 # Neural network minimum learning rate for CLR
NN_MAX_LR = 1e-2 # Neural network maximum learning rate for CLR
CLR_SS_COEFF = 4
BATCH_SIZE = 128 # Batch size for training neural network
EPOCHS = 100 # Maximum number of training epochs
CONV_REG = 0.001 # L2 regularization term for Conv2D layers
DENSE_REG = 0.001 # L2 regularization term for Dense layers
VAL_SPLIT = 0.20 # Fraction of training data to use for validation
MIN_DELTA = 0.01 # Min amount validation loss must decrease to prevent stopping
PATIENCE = 10 # Number of epochs of stagnation before stopping training
POLICY_LOSS_WEIGHT = 1.0 # Weighting given to policy head loss
VALUE_LOSS_WEIGHT = 1.0 # Weighting given to value head loss
# Tournament Parameters
TOURNEY_GAMES = 10 # Number of games in tournament between NNs


# %% Set MCTS Parameters
if TRAINING_ITERATION == 0: # Use random rollouts to generate first dataset
    NEURAL_NET = False
    NUM_TRAINING_GAMES = 200
    CONSTRAINT = 'rollout' # Constraint can be 'rollout' or 'time'
    BUDGET = 400 # Maximum number of rollouts or time in seconds
    MULTIPROC = False # Enable multiprocessing
else: 
    NEURAL_NET = True
    NUM_TRAINING_GAMES = 200
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
    nn = create_nn(conv_reg=CONV_REG, dense_reg=DENSE_REG)
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
    

# %% SELF-PLAY STAGE
# Generate training data
# chk_data = generate_Checkers_data(NUM_TRAINING_GAMES, TRAINING_ITERATION,
#                                   TERMINATE_CNT, **mcts_kwargs)
# training_data, data_fn = chk_data.generate_data()


# %% TRAINING STAGE
# Load existing training data
data_fn = 'data/training_data/Checkers_Data1_16-Jan-2021(21:18:53).pkl'
training_data = load_training_data(data_fn) 
data_fn = 'data/training_data/Checkers_Data1_15-Jan-2021(18:32:57).pkl'
training_data.extend(load_training_data(data_fn)) 
data_fn = 'data/training_data/Checkers_Data0_12-Jan-2021(09:51:47).pkl'
training_data.extend(load_training_data(data_fn))

# Train NN
# run_lr_finder(training_data) # Comment out once CLR parameters are decided
history, new_nn_fn = train_nn(training_data, nn, val_split=VAL_SPLIT)
plot_filename = plot_history(history, nn)


# %% EVALUATION STAGE
# Enter new and old NNs into tournament
print('Beginning tournament between {} and {}!'.format(new_nn_fn, old_nn_fn))
tourney = tournament_Checkers(new_nn_fn, old_nn_fn, TOURNEY_GAMES, 
                              **tourney_kwargs)
tourney_fn = tourney.start_tournament()
record_training_params()