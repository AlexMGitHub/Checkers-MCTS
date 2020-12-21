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

from MCTS import MCTS
from MCTS import MCTS_Node
import numpy as np
import pickle
from datetime import datetime
from tabulate import tabulate
from keras.models import load_model


class generate_Checkers_data():
    """Class to generate Checkers training data through self-play."""
    def __init__(self, NUM_TRAINING_GAMES, TRAINING_ITERATION, TRUNCATE_CNT=100, **mcts_kwargs):
        """Set trainng parameters and initialize MCTS class."""
        self.NUM_TRAINING_GAMES = NUM_TRAINING_GAMES
        self.TRAINING_ITERATION = TRAINING_ITERATION
        self.TRUNCATE_CNT = TRUNCATE_CNT
        self.game_env = mcts_kwargs['GAME_ENV']
        MCTS(**mcts_kwargs) # Set MCTS parameters
   
    def generate_data(self):
        """Generate Checkers training data for a neural network through 
        self-play.  Plays the user-specified number of games, and returns the 
        data as a list of lists.  Each sub-list contains a game state, a
        probability vector, and the terminal reward of the episode from the 
        perspective of the state's current player.
        """
        game_env = self.game_env
        memory = []
        for _ in range(self.NUM_TRAINING_GAMES):
            print('Beginning game {} of {}!'.format(_+1, self.NUM_TRAINING_GAMES))
            experiences = []
            initial_state = game_env.state
            root_node1 = MCTS_Node(initial_state, parent=None)    
            while not game_env.done: # Game loop
                if game_env.current_player(game_env.state) == 'player1':
                    if game_env.move_count != 0:  # Update P1 root node w/ P2's move
                        root_node1 = MCTS.new_root_node(best_child1)
                    MCTS.begin_tree_search(root_node1)
                    best_child1 = MCTS.best_child(root_node1)
                    game_env.step(best_child1.state)
                    prob_vector = self._create_prob_vector(root_node1)
                    experiences.append([root_node1.state, prob_vector])
                else:
                    if game_env.move_count == 1: # Initialize second player's MCTS node 
                       root_node2 = MCTS_Node(game_env.state, parent=None, 
                                              initial_state=initial_state)
                    else: # Update P2 root node with P1's move
                        root_node2 = MCTS.new_root_node(best_child2)
                    MCTS.begin_tree_search(root_node2)
                    best_child2 = MCTS.best_child(root_node2)
                    game_env.step(best_child2.state)
                    prob_vector = self._create_prob_vector(root_node2)
                    experiences.append([root_node2.state, prob_vector])
                if not game_env.done and game_env.move_count >= self.TRUNCATE_CNT:
                    game_env.done = True
                    state = game_env.state
                    p1_cnt = np.sum(state[0:2])
                    p2_cnt = np.sum(state[2:4])
                    if p1_cnt > p2_cnt:
                        game_env.outcome = 'player1_wins'
                    elif p1_cnt < p2_cnt:
                        game_env.outcome = 'player2_wins'
                    else:
                        game_env.outcome = 'draw'
            prob_vector = np.zeros((256,)) # Terminal state
            experiences.append([game_env.state, prob_vector]) # Terminal state
            experiences = self._add_rewards(experiences, game_env.outcome)
            memory.extend(experiences)
            print('{} after {} moves!'.format(game_env.outcome, game_env.move_count))
            game_env.reset()
        if MCTS.multiproc: 
                MCTS.pool.close()
                MCTS.pool.join()
        self._save_memory(memory, self.TRAINING_ITERATION, 
                          self._create_timestamp())
        return memory

    def _create_prob_vector(self, node):
        """Populate the probability vector used to train the neural network's
        policy head.  Uses the probabilities generated by the MCTS for each 
        child node of the given node.
        """
        prob_vector = np.zeros((256,))
        for child in node.children:
            layer = child.state[14,0,0]
            x = child.state[14,0,1]
            y = child.state[14,0,2]
            if x % 2 == y % 2: raise ValueError('Invalid (x,y) locations for probabilities!')
            if not (6 <= layer <= 13): raise ValueError('Invalid layer for probabilities!')
            idx = (layer-6) * 32 + x * 4
            if y % 2 == 0:
                idx += y / 2
            elif y % 2 == 1:
                idx += (y-1) / 2
            prob_vector[int(idx)] = child.n
        prob_vector /= np.sum(prob_vector)
        if not np.isclose(np.sum(prob_vector), 1): 
            raise ValueError('Probabilities do not sum to 1!')
        return prob_vector

    def _add_rewards(self, experiences, outcome):
        """Backpropagate the reward at the end of the episode to every state.
        This is used to train the value head of the neural network by 
        providing the actual outcome of the game as training data.
        """
        for experience in experiences:
            state = experience[0]
            player = int(state[4,0,0])
            if outcome == 'player1_wins':
                reward = -1 if player == 0 else 1
            elif outcome == 'player2_wins':
                reward = -1 if player == 1 else 1
            elif outcome == 'draw':
                reward = 0
            experience.append(reward)
        return experiences

    def _save_memory(self, memory, iteration, timestamp):
        """Save training data to disk as a Pickle file."""
        filename = 'data/training_data/Checkers_Data' + str(iteration) + '_' \
            + timestamp + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(memory, file)
        
    def _create_timestamp(self):
        """Create timestamp string to be used in filenames."""
        timestamp = datetime.now(tz=None)
        timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
        return timestamp_str
    
   
class tournament_Checkers:
    """Class that pits two neural networks against each other in a Checkers
    tournament and saves the result of the tournament to disk.
    """
    def __init__(self, nn1_fn, nn2_fn, num_games, **mcts_kwargs):
        """Load the neural networks based on their supplied filenames.
        Initialize the MCTS class based on supplied parameters.
        """
        self.nn1_fn = nn1_fn
        self.nn2_fn = nn2_fn
        self.nn1 = load_model(nn1_fn)
        self.nn2 = load_model(nn2_fn)
        self.NUM_GAMES = num_games
        self.game_env = mcts_kwargs['GAME_ENV']
        MCTS(**mcts_kwargs) # Set MCTS parameters
    
    def start_tournament(self):
        """Play a Checker's tournament between two neural networks.  The number
        of games played in the tournament is specified by the user, and each
        neural network will play half of the games as player 1 and half as 
        player 2.  Results of the tournament are written to disk.
        """
        game_env = self.game_env
        game_outcomes = []
        for game_num in range(self.NUM_GAMES):
            print('Starting game #{} of {}!'.format(game_num+1, self.NUM_GAMES))
            if game_num < self.NUM_GAMES // 2:
                p1_nn, p2_nn = self.nn1, self.nn2 # Each network is P1 for half
                p1_fn, p2_fn = self.nn1_fn, self.nn2_fn # of the games played.
            else:
                p1_nn, p2_nn = self.nn2, self.nn1 
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
        self._save_tourney_results(game_outcomes)
        print('Tournament over!  View results in tournament folder!')
        
    def _save_tourney_results(self, game_outcomes):
        """Save the results of the tournament to disk.  The file will contain
        two tables.  The first table is a summary of the tournament results 
        (W/L/D).  The second table lists the outcome of each game in the 
        tournament along with the game's turn count.
        """
        fn1 = game_outcomes[0][1]
        fn2 = game_outcomes[0][2]
        fn1_wins, fn2_wins, draws = 0, 0, 0
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
        filename = 'data/tournament_results/Tournament' + \
            self._create_timestamp() + '.txt'
        with open(filename, 'w') as file:
            file.write(tabulate(summary_table, tablefmt='fancy_grid',
                                headers=summary_headers))
            file.write('\n\n')
            file.write(tabulate(game_outcomes, tablefmt='fancy_grid',
                                headers=headers))
    
    def _create_timestamp(self):
        """Create timestamp string to be used in filenames."""
        timestamp = datetime.now(tz=None)
        timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
        return timestamp_str