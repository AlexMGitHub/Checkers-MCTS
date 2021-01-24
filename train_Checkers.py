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
# 1. The iteration of the training pipeline (start with TRAINING_ITERATION = 0)
# 2. The filename of the previous iteration's trained neural network (NN_FN)
# 3. Various parameters for training the neural network.
# 4. Which phases of the training pipeline to run
#
# Outputs:
# 1. Saves the trained neural networks to /data/model.
# 2. Saves plots of the NN training loss to /data/plots.
#
# Notes:
#
#
###############################################################################
"""

# %% Set training pipeline parameters
TRAINING_ITERATION = 1 # Current training iteration
# NN_FN required if TRAINING_ITERATION > 0
NN_FN = 'data/model/Checkers_Model1_16-Jan-2021(21:50:58).h5'
# NEW_NN_FN required if TRAINING = FALSE and EVALUATION = TRUE
NEW_NN_FN = 'data/model/Checkers_Model2_23-Jan-2021(17:47:37).h5'
SELFPLAY = False     # If True self-play phase will be executed
TRAINING = False    # If True training phase will be executed
EVALUATION = True  # If True evaluation phase will be executed


# %% Imports
import os
from training_pipeline2 import record_params
if SELFPLAY or EVALUATION: # Force Keras to use CPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from training_pipeline2 import generate_Checkers_data
if TRAINING:
    from training_pipeline2 import merge_data
    from training_pipeline2 import load_training_data
    from training_pipeline2 import create_nn
    from training_pipeline2 import save_nn_to_disk
    from training_pipeline2 import run_lr_finder
    from training_pipeline2 import train_nn
    from training_pipeline2 import plot_history
    from training_pipeline2 import create_timestamp
    from keras.models import load_model
if EVALUATION:
    from training_pipeline2 import tournament_Checkers


# %% SELF-PLAY STAGE
# Use random rollouts to generate first dataset
NEURAL_NET = False if TRAINING_ITERATION == 0 else True
          
selfplay_kwargs = {
'TRAINING_ITERATION' : TRAINING_ITERATION,
'NN_FN' : NN_FN,
'NUM_SELFPLAY_GAMES' : 50,
'TERMINATE_CNT' : 160,       # Number of moves before terminating training game
'NUM_CPUS' : 4              # Number of CPUs to use for parallel self-play
}

mcts_kwargs = { # Parameters for MCTS used in generating data
'GAME_ENV' : None,          # Game environment loaded in function for multiprocessing
'UCT_C' : 4,                # Constant C used to calculate UCT value
'CONSTRAINT' : 'rollout',   # Constraint can be 'rollout' or 'time'
'BUDGET' : 200,             # Maximum number of rollouts or time in seconds
'MULTIPROC' : False,        # Enable multiprocessing
'NEURAL_NET' : NEURAL_NET,  # If False uses random rollouts instead of NN
'VERBOSE' : False,          # MCTS prints search start/stop messages if True
'TRAINING' : True,          # True if self-play, False if competitive play
'DIRICHLET_ALPHA' : 1.0,    # Used to add noise to prior probs of actions
'DIRICHLET_EPSILON' : 0.25, # Fraction of noise added to prior probs of actions   
'TEMPERATURE_TAU' : 1.0,    # Initial value of temperature Tau
'TEMPERATURE_DECAY' : 0.1,  # Linear decay of Tau per move
'TEMP_DECAY_DELAY' : 10     # Move count before beginning decay of Tau value
}

if SELFPLAY:
    chk_data = generate_Checkers_data(selfplay_kwargs, mcts_kwargs)
    data_fns = chk_data.generate_data()
    record_params('selfplay', **{**selfplay_kwargs, **mcts_kwargs})


# %% TRAINING STAGE
training_kwargs = {     # Parameters used to train neural network
'TRAINING_ITERATION' : TRAINING_ITERATION,
'NN_BASE_LR' : 5e-5,    # Neural network minimum learning rate for CLR
'NN_MAX_LR' : 1e-2,     # Neural network maximum learning rate for CLR
'CLR_SS_COEFF' : 4,     # CLR step-size coefficient
'BATCH_SIZE' : 128,     # Batch size for training neural network
'EPOCHS' : 100,         # Maximum number of training epochs
'CONV_REG' : 0.001,     # L2 regularization term for Conv2D layers
'DENSE_REG' : 0.001,    # L2 regularization term for Dense layers
'NUM_KERNELS' : 128,    # Number of Conv2D kernels in "body" of NN
'VAL_SPLIT' : 0.20,     # Fraction of training data to use for validation
'MIN_DELTA' : 0.01, # Min amount val loss must decrease to prevent stopping
'PATIENCE' : 10,    # Number of epochs of stagnation before stopping training
'POLICY_LOSS_WEIGHT' : 1.0, # Weighting given to policy head loss
'VALUE_LOSS_WEIGHT' : 1.0   # Weighting given to value head loss
}

FIND_LR = False # Set true to find learning rate range for CLR prior to training
SLIDING_WINDOW = 5 # Number of self-play iterations to include in training data

if TRAINING:
    # Load training data
    if SELFPLAY: 
        training_data = merge_data(data_fns, TRAINING_ITERATION)
    else:
        fns = os.listdir('data/training_data')
        token = 'Data' + str(TRAINING_ITERATION)
        data_fns = [fn for fn in fns if token in fn]
        if len(data_fns) > 1:
            training_data = merge_data(data_fns, TRAINING_ITERATION)
            for fn in data_fns: # Move partial files to trash
                fn_path = 'data/training_data/' + fn        
                fn_dest = os.environ["HOME"]+'/.local/share/Trash/files/' + fn
                os.rename(fn_path, fn_dest)
        # Combine previous iterations of data
        WINDOW_START = max(0, TRAINING_ITERATION + 1 - SLIDING_WINDOW)
        DATA_ITERATIONS = list(range(WINDOW_START, TRAINING_ITERATION+1))
        valid_fns = []
        for data_iter in DATA_ITERATIONS:
            token = 'Data' + str(data_iter)
            valid_fns.extend([fn for fn in fns if token in fn])
        training_data = []
        for idx, fn in enumerate(valid_fns):
            data_path = 'data/training_data/' + fn
            training_data.extend(load_training_data(data_path))
    
    # Load NN model
    if TRAINING_ITERATION == 0: 
        nn = create_nn(**training_kwargs)
        NN_FN = save_nn_to_disk(nn, 0, create_timestamp())
    else:
        nn = load_model(NN_FN)
    
    # Determine LR range or begin training
    if FIND_LR:
        run_lr_finder(training_data, start_lr=1e-8, end_lr=1e-1, 
                      num_epochs=6, **training_kwargs) # Find LR range for CLR   
    else:
        # Train NN
        history, NEW_NN_FN = train_nn(training_data, nn, **training_kwargs)
        plot_filename = plot_history(history, nn, TRAINING_ITERATION)
        training_kwargs['OLD_NN_FN'] = NN_FN
        training_kwargs['NEW_NN_FN'] = NEW_NN_FN
        record_params('training', **training_kwargs)

    
# %% EVALUATION STAGE
tourney_kwargs = {
'TRAINING_ITERATION' : TRAINING_ITERATION,
'OLD_NN_FN' : NN_FN,
'NEW_NN_FN' : NEW_NN_FN,   
'TOURNEY_GAMES' : 2,       # Number of games in tournament between NNs
'NUM_CPUS' : 5              # Number of CPUs to use for parallel tourney play
}

tourney_mcts_kwargs = {     # Parameters for MCTS used in tournament
'NN_FN' : NEW_NN_FN,
'UCT_C' : 4,                # Constant C used to calculate UCT value
'CONSTRAINT' : 'rollout',   # Constraint can be 'rollout' or 'time'
'BUDGET' : 200,             # Maximum number of rollouts or time in seconds
'MULTIPROC' : False,        # Enable multiprocessing
'NEURAL_NET' : True,        # If False uses random rollouts instead of NN
'VERBOSE' : False,          # MCTS prints search start/stop messages if True
'TRAINING' : False,         # True if self-play, False if competitive play
'DIRICHLET_ALPHA' : 1.0,    # Used to add noise to prior probs of actions
'DIRICHLET_EPSILON' : 0.25, # Fraction of noise added to prior probs of actions  
'TEMPERATURE_TAU' : 0,      # Initial value of temperature Tau
'TEMPERATURE_DECAY' : 0,    # Linear decay of Tau per move
'TEMP_DECAY_DELAY' : 0      # Move count before beginning decay of Tau value
}

if EVALUATION:
    print('Beginning tournament between {} and {}!'.format(NEW_NN_FN, NN_FN))
    tourney = tournament_Checkers(tourney_kwargs, tourney_mcts_kwargs)
    tourney_fn = tourney.start_tournament()
    record_params('evaluation', **{**tourney_kwargs, **tourney_mcts_kwargs}) 
