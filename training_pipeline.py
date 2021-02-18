#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# training_pipeline.py
#
# Revision:     1.00
# Date:         11/27/2020
# Author:       Alex
#
# Purpose:      Contains classes to generate Checkers training data and to 
#               create a tournament to compare the performance of two 
#               different trained neural networks.  
#
# Classes:
# 1. generate_Checkers_data()   --Generates Checkers training data through 
#                                 self-play. 
#
# 2. tournament_Checkers()      --Pits two neural networks against each other 
#                                 in a Checkers tournament and saves the 
#                                 result of the tournament to disk.
#
# Notes:
# 1. Training data saved to /data/training_data as a Pickle file.
# 2. Tournament results saved to /data/tournament_results.
# 3. Be sure to set the MCTS kwarg 'TRAINING' to False for competitive play.
#
###############################################################################
"""

from tensorflow.keras.utils import Sequence
from Checkers import Checkers
from CLR.clr_callback import CyclicLR
from LRFinder.keras_callback import LRFinder
import numpy as np
import pickle, os
from datetime import datetime
from tabulate import tabulate
import multiprocessing as mp
import matplotlib.pyplot as plt


# %% Functions
def create_nn(**kwargs):
    """Create a double-headed neural network used to learn to play Checkers."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    creg = l2(kwargs['CONV_REG']) # Conv2D regularization param
    dreg = l2(kwargs['DENSE_REG']) # Dense regularization param
    num_kernels = kwargs['NUM_KERNELS'] # Num of Conv2D kernels in "body" of NN
    policy_loss_weight = kwargs['POLICY_LOSS_WEIGHT']
    value_loss_weight = kwargs['VALUE_LOSS_WEIGHT']
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
                  loss_weights={'policy_head' : policy_loss_weight, 
                                  'value_head' : value_loss_weight}, 
                  optimizer=Adam())
    return model


def train_nn(training_data, neural_network, **kwargs):
    """Trains neural network according to desired parameters."""
    import tensorflow as tf
    # Unpack kwargs
    PATIENCE = kwargs['PATIENCE']
    MIN_DELTA = kwargs['MIN_DELTA']
    VAL_SPLIT = kwargs['VAL_SPLIT']
    TRAINING_ITERATION = kwargs['TRAINING_ITERATION']
    BATCH_SIZE = kwargs['BATCH_SIZE']
    CLR_SS_COEFF = kwargs['CLR_SS_COEFF']
    NN_BASE_LR = kwargs['NN_BASE_LR']
    NN_MAX_LR = kwargs['NN_MAX_LR']
    EPOCHS = kwargs['EPOCHS']
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
    if VAL_SPLIT > 0: # Split data into training and validation sets
        validation_data = training_data[-int(len(training_data)*VAL_SPLIT):]
        del training_data[-int(len(training_data)*VAL_SPLIT):]
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
    from tensorflow.keras import backend as K
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

def plot_history(history, nn, TRAINING_ITERATION):
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

def record_params(phase, **kwargs):
    """Document the parameters used in the training pipeline."""
    if phase == 'selfplay':
        filename = 'data/training_data/Checkers_SelfPlay_Params_' + \
            create_timestamp() + '.txt'
    elif phase == 'training':    
        filename = 'data/model/Checkers_Training_Params_' + \
            create_timestamp() + '.txt'
    elif phase == 'evaluation':
        filename = 'data/tournament_results/Checkers_Evaluation_Params_' + \
            create_timestamp() + '.txt'
    elif phase == 'final':
        filename = 'data/final_eval/Checkers_Final_Evaluation_Params_' + \
            create_timestamp() + '.txt'
    else:
        raise ValueError('Invalid phase!')
    # Write parameters to file
    with open(filename, 'w') as file:
        for key, val in kwargs.items():
            file.write('{} = {}\n'.format(key, val))
    
def run_lr_finder(training_data, start_lr, end_lr, num_epochs, **kwargs):
    """Linearly increase learning rate while training neural network over
    a number of epochs.  Outputs plot of training loss versus training
    iteration used to select the base and maximum learning rates used in CLR.
    """
    BATCH_SIZE = kwargs['BATCH_SIZE']
    np.random.shuffle(training_data) # Randomize order of training data
    model = create_nn(**kwargs)
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

def save_merged_files(memory, iteration, timestamp):
    """Save training data to disk as a Pickle file."""
    filename = 'data/training_data/Checkers_Data' + str(iteration) + '_' \
        + timestamp + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(memory, file)
    return filename

def merge_data(data_fns, iteration):
    """Merge multiple training files into a single file."""
    training_data = []        
    for idx, fn in enumerate(data_fns):
        fn_dir = 'data/training_data/' + fn
        training_data.extend(load_training_data(fn_dir))
    filename = save_merged_files(training_data, iteration, create_timestamp()) 
    return training_data


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


class generate_Checkers_data():
    """Class to generate Checkers training data through self-play."""
    def __init__(self, selfplay_kwargs, mcts_kwargs):
        """Set trainng parameters and initialize MCTS class."""
        self.NUM_SELFPLAY_GAMES = selfplay_kwargs['NUM_SELFPLAY_GAMES']
        self.TRAINING_ITERATION = selfplay_kwargs['TRAINING_ITERATION']
        self.TERMINATE_CNT = selfplay_kwargs['TERMINATE_CNT']
        self.num_cpus = selfplay_kwargs['NUM_CPUS']
        self.nn_fn = selfplay_kwargs['NN_FN']
        self.mcts_kwargs = mcts_kwargs
        MAX_PROCESSORS = mp.cpu_count()
        if self.num_cpus > MAX_PROCESSORS: self.num_cpus = MAX_PROCESSORS

    def generate_data(self):
        """Uses the multiprocessing module to parallelize self-play."""
        if self.num_cpus > 1:
            pool = mp.Pool(self.num_cpus)
            filenames = pool.map(self._generate_data, range(self.num_cpus))
            pool.close()
            pool.join()
        else:
            filenames = self._generate_data()
        return filenames  
        
    def _generate_data(self, process_num=0):
        """Generate Checkers training data for a neural network through 
        self-play.  Plays the user-specified number of games, and returns the 
        data as a list of lists.  Each sub-list contains a game state, a
        probability planes, and the terminal reward of the episode from the 
        perspective of the state's current player.
        """
        np.random.seed()
        from tensorflow.keras.models import load_model
        from MCTS import MCTS
        from MCTS import MCTS_Node
        game_env = Checkers(load_model(self.nn_fn))
        self.mcts_kwargs['GAME_ENV'] = game_env
        MCTS(**self.mcts_kwargs) # Set MCTS parameters
        memory = []
        for _ in range(self.NUM_SELFPLAY_GAMES):
            print('Beginning game {} of {}!'.format(_+1, self.NUM_SELFPLAY_GAMES))
            experiences = []
            initial_state = game_env.state
            root_node1 = MCTS_Node(initial_state, parent=None)
            terminated_game = False
            parent_player = 'player2'
            while not game_env.done: # Game loop
                if game_env.current_player(game_env.state) == 'player1':
                    if game_env.move_count != 0:  # Update P1 root node w/ P2's move
                        parent_player = MCTS.current_player(game_env.history[-2])
                        root_node1 = MCTS.new_root_node(best_child1)
                    MCTS.begin_tree_search(root_node1)
                    best_child1 = MCTS.best_child(root_node1)
                    game_env.step(best_child1.state)
                    prob_planes = self._create_prob_planes(root_node1)
                    if parent_player != root_node1.player:
                        qval = -root_node1.q  
                    else:
                        qval = root_node1.q
                    experiences.append([root_node1.state, prob_planes, qval])
                else:
                    if game_env.move_count == 1: # Initialize second player's MCTS node 
                       root_node2 = MCTS_Node(game_env.state, parent=None, 
                                              initial_state=initial_state)
                       parent_player = 'player1'
                    else: # Update P2 root node with P1's move
                        parent_player = MCTS.current_player(game_env.history[-2])
                        root_node2 = MCTS.new_root_node(best_child2)
                    MCTS.begin_tree_search(root_node2)
                    best_child2 = MCTS.best_child(root_node2)
                    game_env.step(best_child2.state)
                    prob_planes = self._create_prob_planes(root_node2)
                    if parent_player != root_node2.player:
                        qval = -root_node2.q  
                    else:
                        qval = root_node2.q
                    experiences.append([root_node2.state, prob_planes, qval])
                if not game_env.done and game_env.move_count >= self.TERMINATE_CNT:
                    terminated_game = True
                    game_env.done = True
                    state = game_env.state
                    p1_cnt = np.sum(state[0:2])
                    p2_cnt = np.sum(state[2:4])
                    p1_king_cnt = np.sum(state[1])
                    p2_king_cnt = np.sum(state[3])
                    if p1_cnt > p2_cnt:
                        game_env.outcome = 'player1_wins'
                    elif p1_cnt < p2_cnt:
                        game_env.outcome = 'player2_wins'
                    else:
                        if p1_king_cnt > p2_king_cnt:
                            game_env.outcome = 'player1_wins'
                        elif p1_king_cnt < p2_king_cnt:
                            game_env.outcome = 'player2_wins'
                        else:
                            game_env.outcome = 'draw'
            if not terminated_game: # Include terminal state
                prob_planes = np.zeros((8,8,8))
                node_q = 0 if game_env.outcome == 'draw' else -1
                experiences.append([game_env.state, prob_planes, node_q])
            experiences = self._add_rewards(experiences, game_env.outcome)
            memory.extend(experiences)
            print('{} after {} moves!'.format(game_env.outcome, game_env.move_count))
            game_env.reset()
        if MCTS.multiproc: 
                MCTS.pool.close()
                MCTS.pool.join()
        filename = self._save_memory(memory, self.TRAINING_ITERATION, 
                          self._create_timestamp(), process_num)
        return filename

    def _create_prob_planes(self, node):
        """Populate the probability planes used to train the neural network's
        policy head.  Uses the probabilities generated by the MCTS for each 
        child node of the given node.
        """
        prob_planes = np.zeros((8,8,8))
        for child in node.children:
            layer = int(child.state[14,0,0] - 6)
            x = int(child.state[14,0,1])
            y = int(child.state[14,0,2])
            if x % 2 == y % 2: raise ValueError('Invalid (x,y) locations for probabilities!')
            if not (0 <= layer <= 7): raise ValueError('Invalid layer for probabilities!')
            prob_planes[layer, x, y] = child.n
        prob_planes /= np.sum(prob_planes)
        if not np.isclose(np.sum(prob_planes), 1): 
            raise ValueError('Probabilities do not sum to 1!')
        return prob_planes

    def _add_rewards(self, experiences, outcome):
        """Include a reward with every state based on the outcome of the 
        episode.  This is used to train the value head of the neural network by 
        providing the actual outcome of the game as training data.  Note that
        the rewards are not reversed like in the MCTS.
        """
        for experience in experiences:
            state = experience[0]
            player = int(state[4,0,0])
            if outcome == 'player1_wins':
                reward = 1 if player == 0 else -1
            elif outcome == 'player2_wins':
                reward = 1 if player == 1 else -1
            elif outcome == 'draw':
                reward = 0
            experience.append(reward)
        return experiences

    def _save_memory(self, memory, iteration, timestamp, process_num):
        """Save training data to disk as a Pickle file."""
        filename = 'data/training_data/Checkers_Data' + str(iteration) + '_' \
            + timestamp + '_P' + str(process_num) + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(memory, file)
        return filename
        
    def _create_timestamp(self):
        """Create timestamp string to be used in filenames."""
        timestamp = datetime.now(tz=None)
        timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
        return timestamp_str
    
   
class tournament_Checkers:
    """Class that pits two neural networks against each other in a Checkers
    tournament and saves the result of the tournament to disk.
    """
    def __init__(self, tourney_kwargs, mcts_kwargs):
        """Load the neural networks based on their supplied filenames.
        Initialize the MCTS class based on supplied parameters.
        """
        self.nn1_fn = tourney_kwargs['NEW_NN_FN']
        self.nn2_fn = tourney_kwargs['OLD_NN_FN']
        self.NUM_GAMES = tourney_kwargs['TOURNEY_GAMES']
        self.mcts_kwargs = mcts_kwargs
        self.num_cpus = tourney_kwargs['NUM_CPUS']
        MAX_PROCESSORS = mp.cpu_count()
        if self.num_cpus > MAX_PROCESSORS: self.num_cpus = MAX_PROCESSORS
    
    def start_tournament(self):
        """Uses the multiprocessing module to parallelize tournament play."""
        if self.num_cpus > 1:
            pool = mp.Pool(self.num_cpus)
            outcomes = pool.map(self._start_tournament, range(self.num_cpus))
            pool.close()
            pool.join()
            self.outcomes = outcomes
            game_outcomes = []
            for outcome in outcomes:
                game_outcomes.extend(outcome)
        else:
            game_outcomes = self._start_tournament()
        filename = self._save_tourney_results(game_outcomes)    
        print('Tournament over!  View results in tournament folder!')
        return filename
    
    def _start_tournament(self, process_num=0):
        """Play a Checker's tournament between two neural networks.  The number
        of games played in the tournament is specified by the user, and each
        neural network will play half of the games as player 1 and half as 
        player 2.  Results of the tournament are written to disk.
        """
        np.random.seed()
        from tensorflow.keras.models import load_model
        from MCTS import MCTS
        from MCTS import MCTS_Node
        nn1 = load_model(self.nn1_fn)
        nn2 = load_model(self.nn2_fn)
        game_env = Checkers(nn1)
        self.mcts_kwargs['GAME_ENV'] = game_env
        MCTS(**self.mcts_kwargs) # Set MCTS parameters
        game_outcomes = []
        for game_num in range(self.NUM_GAMES):
            print('Starting game #{} of {}!'.format(game_num+1, self.NUM_GAMES))
            if game_num < self.NUM_GAMES // 2:
                p1_nn, p2_nn = nn1, nn2 # Each network is P1 for half
                p1_fn, p2_fn = self.nn1_fn, self.nn2_fn # of the games played.
            else:
                p1_nn, p2_nn = nn2, nn1 
                p1_fn, p2_fn = self.nn2_fn, self.nn1_fn
            game_env.neural_net = p1_nn
            initial_state = game_env.state
            root_node1 = MCTS_Node(initial_state, parent=None)    
            while not game_env.done: # Game loop
                if game_env.current_player(game_env.state) == 'player1':
                    if game_env.move_count != 0:  # Update P1 root node w/ P2's move
                        root_node1 = MCTS.new_root_node(best_child1)
                    game_env.neural_net = p1_nn # Use P1's neural network
                    MCTS.begin_tree_search(root_node1)
                    best_child1 = MCTS.best_child(root_node1)
                    game_env.step(best_child1.state)
                else:
                    if game_env.move_count == 1: # Initialize second player's MCTS node 
                       root_node2 = MCTS_Node(game_env.state, parent=None, 
                                              initial_state=initial_state)
                    else: # Update P2 root node with P1's move
                        root_node2 = MCTS.new_root_node(best_child2)
                    game_env.neural_net = p2_nn # Use P2's neural network
                    MCTS.begin_tree_search(root_node2)
                    best_child2 = MCTS.best_child(root_node2)
                    game_env.step(best_child2.state)
            p1_fn_only = p1_fn.replace('data/model/','')
            p2_fn_only = p2_fn.replace('data/model/','')
            game_outcomes.append([game_num+1, p1_fn_only, p2_fn_only, 
                                  game_env.outcome, game_env.move_count])
            print('{} after {} moves!'.format(game_env.outcome, game_env.move_count))
            game_env.reset()
        if MCTS.multiproc: 
                MCTS.pool.close()
                MCTS.pool.join()
        return game_outcomes
        
    def _save_tourney_results(self, game_outcomes):
        """Save the results of the tournament to disk.  The file will contain
        two tables.  The first table is a summary of the tournament results 
        (W/L/D).  The second table lists the outcome of each game in the 
        tournament along with the game's turn count.
        """
        fn1 = game_outcomes[0][1]
        fn2 = game_outcomes[0][2]
        fn1_wins, fn2_wins, draws = 0, 0, 0
        for idx, outcome_list in enumerate(game_outcomes):
            outcome_list[0] = idx+1 # Renumber games
        for game_num, p1_fn, p2_fn, outcome, move_count in game_outcomes:
            if outcome == 'player1_wins':
                if p1_fn == fn1: fn1_wins += 1
                if p1_fn == fn2: fn2_wins += 1                        
            elif outcome == 'player2_wins':                    
                if p2_fn == fn1: fn1_wins += 1
                if p2_fn == fn2: fn2_wins += 1                        
            elif outcome == 'draw':
                draws += 1
        fn1_wld = str(fn1_wins) + '/' + str(fn2_wins) + '/' + str(draws)
        fn2_wld = str(fn2_wins) + '/' + str(fn1_wins) + '/' + str(draws)
        summary_table = [[fn1, fn1_wld],[fn2, fn2_wld]]
        summary_headers = ['Neural Network', 'Wins/Losses/Draws']
        headers = ['Game Number', 'Player 1', 'Player 2', 'Outcome', 'Turn Count']
        filename = 'data/tournament_results/Tournament_' + \
            self._create_timestamp() + '.txt'
        with open(filename, 'w') as file:
            file.write(tabulate(summary_table, tablefmt='fancy_grid',
                                headers=summary_headers))
            file.write('\n\n')
            file.write(tabulate(game_outcomes, tablefmt='fancy_grid',
                                headers=headers))
        return filename
    
    def _create_timestamp(self):
        """Create timestamp string to be used in filenames."""
        timestamp = datetime.now(tz=None)
        timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
        return timestamp_str


class final_evaluation():
    """Class that illustrates the relative improvement among trained models
    produced during the training pipeline.  The selected models are placed in
    a tournament where each model plays every other model twice.  The results
    are tabulated and plotted.
    """
    def __init__(self, model_iter_list, tourney_kwargs, mcts_kwargs):
        """Accepts a list of training iterations and finds the corresponding 
        trained models in the data/model directory.  Also accepts tournament
        parameters in the form of dictionaries.
        """
        self.model_iter_list = model_iter_list
        self.model_fn_list = []
        self.tourney_kwargs = tourney_kwargs
        self.mcts_kwargs = mcts_kwargs
        self.num_cpus = tourney_kwargs['NUM_CPUS']
        self.tourney_kwargs['TOURNEY_GAMES'] = 2
        fns = os.listdir('data/model')
        for iter_num in model_iter_list:
            for fn in fns:
                model_fn = 'Model' + str(iter_num) + '_'
                if model_fn in fn and '.h5' in fn:
                    self.model_fn_list.append(fn)
                    break
        if len(self.model_fn_list) != len(self.model_iter_list):
            raise ValueError('Model(s) not found!')
        self.table = np.zeros((len(model_iter_list),len(model_iter_list)))
        self.game_outcomes = []
    
    def start_evaluation(self, num_cpus):
        """Uses the multiprocessing module to parallelize tournament play."""
        model_fn_list = self.model_fn_list.copy()
        self.num_cpus = num_cpus
        for _ in range(len(self.model_fn_list)-1):
            game_outcomes = []
            new_nn_fn = model_fn_list.pop()
            self.tourney_kwargs['NEW_NN_FN'] = 'data/model/' + new_nn_fn
            partial_model_fn_list = model_fn_list.copy()
            while len(partial_model_fn_list) > 0:
                if len(partial_model_fn_list) <= self.num_cpus:
                    num_cpus = len(partial_model_fn_list)
                    pool_fn_list = partial_model_fn_list.copy()
                    del partial_model_fn_list[:]
                else:
                    num_cpus = self.num_cpus
                    pool_fn_list = partial_model_fn_list[-num_cpus:]
                    del partial_model_fn_list[-num_cpus:]
                pool = mp.Pool(num_cpus)
                outcomes = pool.map(self._wrapper_func, pool_fn_list)
                pool.close()
                pool.join()
                for outcome in outcomes:
                    game_outcomes.extend(outcome)
            self.game_outcomes.append(game_outcomes)
        self._parse_tourney_results()    
        print('Final evaluation over!  View results in final_eval folder!')

    def _wrapper_func(self, nn_fn):
        """Wrapper function used by the multiprocessing."""
        tourney_kwargs = self.tourney_kwargs.copy()
        tourney_mcts_kwargs = self.mcts_kwargs.copy()
        tourney_kwargs['NUM_CPUS'] = 1
        tourney_kwargs['OLD_NN_FN'] = 'data/model/' + nn_fn
        tourney_mcts_kwargs['NN_FN'] = tourney_kwargs['NEW_NN_FN']
        print('Beginning tournament between {} and {}!'
              .format(tourney_kwargs['NEW_NN_FN'], tourney_kwargs['OLD_NN_FN']))
        tourney = tournament_Checkers(tourney_kwargs, tourney_mcts_kwargs)
        return tourney._start_tournament()
        
    def _parse_tourney_results(self):
        """Save the results of the tournament to disk.  There will be two 
        outputs saved to the data/final_eval directory: a table of the 
        tournament results and a plot of the models' total score.
        """
        for game_outcomes in self.game_outcomes:
            for game_num, p1_fn, p2_fn, outcome, move_count in game_outcomes:
                p1_idx = self.model_fn_list.index(p1_fn)
                p2_idx = self.model_fn_list.index(p2_fn)
                if outcome == 'player1_wins':
                    self.table[p1_idx, p2_idx] += 1
                    self.table[p2_idx, p1_idx] -= 1
                elif outcome == 'player2_wins':                    
                    self.table[p1_idx, p2_idx] -= 1
                    self.table[p2_idx, p1_idx] += 1                        
        model_scores = np.sum(self.table, axis=1)
        self._plot_model_scores(model_scores)
        col_headers = self.model_iter_list + ['Total']
        table = np.hstack((self.table, np.transpose(model_scores[np.newaxis])))
        filename = 'data/final_eval/Checkers_Final_Evaluation_' + \
                    self._create_timestamp() + '.txt'
        with open(filename, 'w') as file:
            file.write(tabulate(table, headers=col_headers, 
                                showindex=self.model_iter_list, 
                                tablefmt='fancy_grid'))
    
    def _plot_model_scores(self, model_scores):
        """Saves plot of final eval points versus model iteration to disk."""
        plt.plot(self.model_iter_list, model_scores, marker='o')
        plt.title('Final Evaluation')
        plt.ylabel('Points')
        plt.xlabel('Model Iteration Number')
        plt.grid()
        filename = 'data/final_eval/Checkers_Final_Evaluation_' + \
                    self._create_timestamp() + '.png'
        plt.draw()
        fig1 = plt.gcf()
        fig1.set_dpi(200)
        fig1.savefig(filename)
        plt.show()
        plt.close()
        return filename
            
    def _create_timestamp(self):
        """Create timestamp string to be used in filenames."""
        timestamp = datetime.now(tz=None)
        timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
        return timestamp_str