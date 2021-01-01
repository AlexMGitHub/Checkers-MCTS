#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# play_Checkers.py
#
# Revision:     1.00
# Date:         11/15/2020
# Author:       Alex
#
# Purpose:      Plays a demonstration game of Checkers using the Monte Carlo
#               Tree Search algorithm. 
#
# Inputs:
# 1. MCTS parameters, e.g. the computational constraints and UCT constant.
# 2. Player selection - human vs. MCTS algorithm or algorithm vs. algorithm.
#
# Outputs:
# 1. Text representations of the Tic-Tac-Toe game board and the MCTS tree.
# 2. An optional print out of the MCTS tree after each player's move.
#
# Notes:
# 1. Run this module to see a demonstration game of Checkers played using 
#    the MCTS algorithm.
#
###############################################################################
"""
# %% Imports
from Checkers import Checkers
from Checkers import Checkers_GUI
from MCTS import MCTS
from MCTS import MCTS_Node
import numpy as np
from keras.models import load_model


# %% Functions
def get_human_input():
    """Print a list of legal next moves for the human player, and return
    the player's selection.  The player will be presented with move options
    in the form of two coordinates that represent the start and end location
    of the piece to be moved.
    """
    while True:
        legal_next_states = game_env.legal_next_states
        moves_list = states_to_piece_positions(legal_next_states)
        for idx, move in enumerate(moves_list):
            print('Option #{}: {} to {}'.format(idx+1, move[0], move[1]))
        move_idx = int(input('Enter option number: ')) - 1
        if move_idx in range(len(moves_list)):
            game_env.step(legal_next_states[move_idx])
            return legal_next_states[move_idx]
        else:
            print('Invalid selection!  Try again!')

def states_to_piece_positions(next_states):
    """Given a list of next states, produce a list of two coordinates for each 
    possible next state.  The first coordinate will be the location of the 
    piece that was moved, and the second coordinate will be the location that 
    the piece moved to.
    """
    moves_list = []
    state = game_env.state
    board = state[0] + 2*state[1] + 3*state[2] + 4*state[3]
    for nstate in next_states:
        nboard = nstate[0] + 2*nstate[1] + 3*nstate[2] + 4*nstate[3]
        board_diff = board - nboard
        xnew, ynew = np.where(board_diff < 0)
        xnew, ynew = xnew[0], ynew[0]
        new_val = abs(nboard[xnew,ynew])
        xold, yold = np.where(board_diff == new_val)
        try:
            xold, yold = xold[0], yold[0]
        except IndexError: # Man promoted to king
            new_val -= 1 # Value of man is 1 less than king
            xold, yold = np.where(board_diff == new_val)
            xold, yold = xold[0], yold[0]
        moves_list.append([(xold+1,yold+1),(xnew+1,ynew+1)])
    return moves_list


# %% Initialize game environment and MCTS class
# Set MCTS parameters
UCT_C = 1/(2**0.5) # Constant C used to calculate UCT value
CONSTRAINT = 'rollout' # Constraint can be 'rollout' or 'time'
BUDGET = 200 # Maximum number of rollouts or time in seconds
MULTIPROC = False # Enable multiprocessing
NEURAL_NET = True # Use random rollouts if False
VERBOSE = True # MCTS prints search start/stop messages if True
TRAINING = False # True if training NN, False if competitive play
DIRICHLET_ALPHA = 3.6 # Used to add noise to prior probabilities of actions
DIRICHLET_EPSILON = 0.25 # Used to add noise to prior probabilities of actions
TEMPERATURE_TAU = 1.0 # Initial value of temperature Tau
TEMPERATURE_DECAY = 0.1 # Linear decay of Tau per move
TEMP_DECAY_DELAY = 10 # Move count before beginning decay of Tau value
# Initialize game environment
GUI = True # Enable Pygame GUI
if NEURAL_NET:
    nn = load_model('data/model/Checkers_Model1_26-Dec-2020(14:44:12).h5')
    game_env = Checkers(nn)
else:
    game_env = Checkers(neural_net=None)
initial_state = game_env.state
game_env.print_board()
if GUI: checker_gui = Checkers_GUI(game_env)

mcts_kwargs = {
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
MCTS(**mcts_kwargs)

# Choose whether to play against the MCTS or to pit them against each other
human_player1 = False # Set true to play against the MCTS algorithm as player 1
human_player2 = False # Or choose player 2
if human_player1 and human_player2: human_player2 = False
human_player_idx = 0 if human_player1 else 1
if not human_player1:
    root_node1 = MCTS_Node(initial_state, parent=None)    

print_trees = True # Choose whether to print root node's tree after every move
tree_depth = 1 # Number of layers of tree to print (warning: expands quickly!)


# %% Game loop
while not game_env.done:
    if game_env.current_player(game_env.state) == 'player1':
        if human_player1: 
            human_move = get_human_input()
        else: # MCTS plays as player 1
            if game_env.move_count != 0:  # Update P1 root node w/ P2's move
                root_node1 = MCTS.new_root_node(best_child1)
            MCTS.begin_tree_search(root_node1)
            best_child1 = MCTS.best_child(root_node1)
            game_env.step(best_child1.state)
            if print_trees: MCTS.print_tree(root_node1,tree_depth)
            game_env.print_board()
            if GUI: checker_gui.render(root_node1, best_child1)      
    else:
        if human_player2: 
            human_move = get_human_input()
        else: # MCTS plays as player 2
            if game_env.move_count == 1: # Initialize second player's MCTS node 
               root_node2 = MCTS_Node(game_env.state, parent=None, 
                                      initial_state=initial_state)
            else: # Update P2 root node with P1's move
                root_node2 = MCTS.new_root_node(best_child2)
            MCTS.begin_tree_search(root_node2)
            best_child2 = MCTS.best_child(root_node2)
            game_env.step(best_child2.state)
            if print_trees: MCTS.print_tree(root_node2,tree_depth)
            game_env.print_board()
            if GUI: checker_gui.render(root_node2, best_child2)      
        
# Housekeeping after game over
if GUI: 
    input('Press enter to continue...')
    checker_gui.close_gui()
if MULTIPROC: 
    MCTS.pool.close()
    MCTS.pool.join()
