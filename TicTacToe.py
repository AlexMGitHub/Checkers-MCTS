#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# TicTacToe.py
#
# Revision:     1.00
# Date:         11/07/2020
# Author:       Alex
#
# Purpose:      Contains all functions necessary to implement the Tic-Tac-Toe 
#               game environment.
#
# Notes:
# 1. Run this module to see a demonstration game of Tic-Tac-Toe.
#
###############################################################################
"""

import numpy as np
from copy import deepcopy
from tabulate import tabulate


class TicTacToe:
    """Class to represent a game of Tic-Tac-Toe."""
    def __init__(self):
        """Initialize the class with an empty board.  Get a list of valid
        (legal) next moves.  Moves are not explicitly represented; instead 
        possible next states of the game board are generated.  The player's
        move is implied by selecting one of these possible next states.
        """
        self.state = np.zeros((3,3,3), dtype=float)
        self.history = [self.state]
        self.legal_next_states = self.get_legal_next_states(self.history)
        self.player1_mark = 'X'
        self.player2_mark = 'O'
        self.done = False
        self.outcome = None
        self.move_count = 0
        
    def step(self, next_state):
        """Execute the player's (legal) move.  Check to see if the game
        has ended, and update the list of legal next moves.
        """
        if any((next_state == x).all() for x in self.legal_next_states):
            self.state = next_state
            self.history.append(self.state)
            self.done, self.outcome = self.determine_outcome(self.history)
            self.legal_next_states = self.get_legal_next_states(self.history)
            self.move_count += 1
            return self.state, self.outcome, self.done
        else:
            raise ValueError('Illegal next state (invalid move)!')
     
    def get_legal_next_states(self, history):
        """If the game is not done, return a list of legal next moves given
        a board state as input.  The next moves are actually board states;
        the move to achieve those states is implied.
        """
        done, outcome = self.determine_outcome(history)
        if done == True: return [] # Game over
        state = history[-1]
        player = int(state[2,0,0])
        board = state[0] + state[1] # Combine player 1 and 2's pieces
        x_coords, y_coords = np.where(board == 0) # Find empty squares
        legal_next_states = []
        for x, y in zip(x_coords, y_coords):
            next_state = deepcopy(state)
            next_state[player, x, y] = 1 # Player chooses available empty square
            next_state[2] = 1 - player # Toggle player
            legal_next_states.append(next_state)
        return legal_next_states
    
    def determine_outcome(self, history):
        """Given a board state as input, determine if the game is over or not.  
        If the game is over, determine the winner (or if the game is a draw).
        """
        state = history[-1]
        # Check if Player 1 won (3 X's in a row)
        check_p1_cols = np.max(np.sum(state[0], axis=0))
        check_p1_rows = np.max(np.sum(state[0], axis=1))
        check_p1_diag1 = np.sum(state[0].diagonal())
        check_p1_diag2 = np.sum(np.fliplr(state[0]).diagonal())
        # Check if Player 2 won (3 O's in a row)                        
        check_p2_cols = np.max(np.sum(state[1], axis=0))
        check_p2_rows = np.max(np.sum(state[1], axis=1))
        check_p2_diag1 = np.sum(state[1].diagonal())
        check_p2_diag2 = np.sum(np.fliplr(state[1]).diagonal())
        # Check to see if all squares are filled
        num_squares_filled = np.sum(state[0] + state[1])
        # Return done flag and outcome of game
        if np.max([check_p1_cols, check_p1_rows, check_p1_diag1, check_p1_diag2]) == 3:
            done = True
            outcome = 'player1_wins'
        elif np.max([check_p2_cols, check_p2_rows, check_p2_diag1, check_p2_diag2]) == 3:
            done = True
            outcome = 'player2_wins'
        elif num_squares_filled == 9:
            done = True
            outcome = 'draw'
        else: 
            done = False # Game is still in progress
            outcome = None
        return done, outcome
        
    def print_board(self):
        """Print a visual representation of the current game state to the 
        console."""
        player = int(self.state[2,0,0])
        player_mark = self.player1_mark if player == 0 else self.player2_mark 
        board_pieces = self.state[0] - self.state[1]
        board_table = []
        for row in board_pieces:
            row_list = []
            for square in row:
                if square == 1: row_list.append(self.player1_mark)
                if square == -1: row_list.append(self.player2_mark)
                if square == 0: row_list.append('.')
            board_table.append(row_list)
        print(tabulate(board_table, tablefmt='fancy_grid'))
        if not self.done:
            print('It\'s now Player {}\'s turn ({})'.format(player+1, player_mark))
        else:
            print('Game over! The outcome is: {}'.format(self.outcome))
                
    def current_player(self, state):
        """Return which player's turn it is for a given input state."""
        player = int(state[2,0,0])
        if player == 0:
            return 'player1'
        else:
            return 'player2'
    
    def reset(self):
        """Reset the board so it's ready for the next game."""
        self.state = np.zeros((3,3,3), dtype=float)
        self.legal_next_states = self.get_legal_next_states(self.state)
        self.done = False
        self.outcome = None
        self.move_count = 0
        self.history = []
    

def test_game():
    """Test function to validate functionality of TicTacToe class.
    Plays a game of Tic-Tac-Toe by randomly selecting moves for both players.
    """
    ttt = TicTacToe()
    ttt.print_board()
    while not ttt.done:
        legal_next_states = ttt.get_legal_next_states(ttt.state)
        move_idx = np.random.randint(0,len(legal_next_states))
        ttt.step(legal_next_states[move_idx])
        ttt.print_board()
    
    
if __name__ == '__main__':
    test_game()