#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# play_TTT.py
#
# Revision:     1.00
# Date:         11/07/2020
# Author:       Alex
#
# Purpose:      Plays a demonstration game of Tic-Tac-Toe using the Monte Carlo
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
# 1. Run this module to see a demonstration game of Tic-Tac-Toe played using 
#    the MCTS algorithm.
#
###############################################################################
"""
# %% Imports
from TicTacToe import TicTacToe
from MCTS import MCTS
from MCTS import MCTS_Node


# %% Functions
def get_human_input():
    """Print a list of legal next states for the human player, and return
    the player's selection.
    """
    legal_next_states = game_env.legal_next_states
    for idx, state in enumerate(legal_next_states):
        print(state[human_player_idx], '\t', idx, '\n')
    move_idx = int(input('Enter move index: '))
    game_env.step(legal_next_states[move_idx])
    return legal_next_states[move_idx]


# %% Initialize game environment and MCTS class
game_env = TicTacToe()
initial_state = game_env.state
game_env.print_board()

# Set MCTS parameters
mcts_kwargs = {     # Parameters for MCTS used in tournament
'GAME_ENV' : game_env,
'UCT_C' : 4,                # Constant C used to calculate UCT value
'CONSTRAINT' : 'rollout',   # Constraint can be 'rollout' or 'time'
'BUDGET' : 2000,            # Maximum number of rollouts or time in seconds
'MULTIPROC' : False,        # Enable multiprocessing
'NEURAL_NET' : False,       # If False uses random rollouts instead of NN
'VERBOSE' : True,           # MCTS prints search start/stop messages if True
'TRAINING' : False,         # True if self-play, False if competitive play
'DIRICHLET_ALPHA' : 1.0,    # Used to add noise to prior probs of actions
'DIRICHLET_EPSILON' : 0.25, # Fraction of noise added to prior probs of actions  
'TEMPERATURE_TAU' : 0,      # Initial value of temperature Tau
'TEMPERATURE_DECAY' : 0,    # Linear decay of Tau per move
'TEMP_DECAY_DELAY' : 0      # Move count before beginning decay of Tau value
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