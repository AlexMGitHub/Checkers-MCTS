#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# Checkers.py
#
# Revision:     1.00
# Date:         11/11/2020
# Author:       Alex
#
# Purpose:      Contains all functions necessary to implement the Checkers
#               game environment.
#
# Notes:
# 1. Run this module to see a demonstration game of Checkers.
# 2. Rules are according to the World Checkers Draughts Federation (WCDF).
# 3. https://www.wcdf.net/rules.htm
#
###############################################################################
"""

import numpy as np
from copy import deepcopy
from tabulate import tabulate
import pygame, time


class Checkers:
    """Class to represent a game of Checkers."""
    def __init__(self, neural_net=None):
        """Initialize the class with the pieces in their starting positions.  
        Get a list of valid (legal) next moves.  Moves are not explicitly 
        represented; instead possible next states of the game board are 
        generated.  The player's move is implied by selecting one of these 
        possible next states.
        
        The game state is a 3 dimensional NumPy array of 5 8x8 arrays.
        These 8x8 arrays represent the 8x8 game board, where:
        0. Array 0 represents the locations of player 1's uncrowned men.
        1. Array 1 represents the locations of player 1's kings.
        2. Array 2 represents the locations of player 2's uncrowned men.
        3. Array 3 represents the locations of player 2's kings.
        4. Array 4 indicates the current player (all 0s for P1, all 1s for P2)
        5. Array 5 is the draw timer; counts in increments of 1/80
        6. Arrays 6, 7, 8, 9 are normal moves (UL, UR, BL, BR)
        7. Arrays 10, 11, 12, 13 are jumps (UL, UR, BL, BR)
        8. Array 14 contains the indices of the parent state's action
        """
        self.state = np.zeros((15,8,8), dtype=float)
        self.init_board()
        self.history = [self.state]
        self.legal_next_states = self.get_legal_next_states(self.history)
        self.move_count = 0
        self.done = False
        self.outcome = None
        self.player1_man = 'x'
        self.player1_king = u'\u0416'
        self.player2_man = 'o'
        self.player2_king = u'\u01D1'
        self.neural_net = neural_net
        
    def step(self, next_state):
        """Execute the player's (legal) move.  Check to see if the game
        has ended, and update the list of legal next moves.
        """
        if any((next_state[:5] == x[:5]).all() for x in self.legal_next_states):
            self.state = next_state
            self.history.append(self.state)
            self.legal_next_states = self._check_moves(self.history)
            self.done, self.outcome = self.determine_outcome(self.history,
                                         legal_moves=self.legal_next_states)
            self.move_count += 1
            return self.state, self.outcome, self.done
        else:
            raise ValueError('Illegal next state (invalid move)!')
     
    def get_legal_next_states(self, history):
        """If the game is not done, return a list of legal next moves given
        a history of moves as input.  The next moves are actually board states;
        the move to achieve those states is implied.
        
        This function calls determine_outcome() which also must check 
        the legal next states in order to determine the outcome of the game.  
        Redundant computation can be avoided by checking the legal next states 
        here first and then passing them as an optional argument to 
        determine_outcome().
        """
        legal_next_states = self._check_moves(history)
        done, outcome = self.determine_outcome(history, 
                                               legal_moves=legal_next_states)
        if done == True: return [] # Game over
        return legal_next_states
    
    def _check_moves(self, history):
        """Method intended for internal use.  Creates a list of the locations 
        of all of the pieces on the board divided up into four categories (P1's 
        men, P1's kings, P2's men, and P2's kings).
        
        Checks for all possible ordinary moves of men and kings for the current 
        player only.  Calls two other internal methods, _check_jumps() 
        and _check_king_jumps() to determine if there are jumps possible for
        the player's men and kings, respectively.  Per the rules jumps are 
        mandatory moves, and so if jump moves exist they are returned by the
        function instead of the ordinary moves.
        
        The function also determines if an ordinary move results in a man 
        reaching King's Row, and kings the man if so.
        """
        state = history[-1]
        player = int(state[4,0,0])
        xman1, yman1 = np.where(state[0] == 1) # Locations of P1's men
        xking1, yking1 = np.where(state[1] == 1) # Locations of P1's kings
        xman2, yman2 = np.where(state[2] == 1) # Locations of P2's men
        xking2, yking2 = np.where(state[3] == 1) # Locations of P2's kings
        piece_locs = [np.column_stack((xman1, yman1)), np.column_stack((xking1, yking1)), 
                      np.column_stack((xman2, yman2)), np.column_stack((xking2, yking2))]
        board = np.sum(state[0:4], axis=0) # All pieces on one 8x8 grid
        idx = player * 2 # State index of player's pieces
        opp_idx = 0 if idx else 2 # State index of opponent's pieces
        fwd = 1 if player == 0 else -1 # Sets forward direction of player's men
        legal_moves = []
        jump_moves = []
        # Get legal moves including jumps for men
        for x, y in piece_locs[idx]: # Men
            if y+1 < 8 and -1 < x+fwd < 8:
                if board[x+fwd,y+1] == 0: # Diagonal-right space open
                    temp_state = deepcopy(state)
                    temp_state[5:] = 0 # Erase NN layers from previous state
                    temp_state[4] = 1 - player # Toggle player
                    temp_state[idx,x,y] = 0 # Piece no longer in prev location
                    if (fwd == 1 and x+fwd == 7) or \
                    (fwd == -1 and x+fwd == 0): # On King's row, king the man
                        temp_state[idx+1,x+fwd,y+1] = 1
                    else: # Not on King's row, man does not become king
                        temp_state[idx,x+fwd,y+1] = 1
                    if fwd == 1:
                        state[9,x,y] = 1 # NN layer representing BR move
                        temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                            9, x, y
                    else:
                        state[7,x,y] = 1 # NN layer representing UR move
                        temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                            7, x, y
                    legal_moves.append(temp_state)
            if y-1 > -1 and -1 < x+fwd < 8:
                if board[x+fwd,y-1] == 0: # Diagonal-left space open
                    temp_state = deepcopy(state)
                    temp_state[5:] = 0 # Erase NN layers from previous state
                    temp_state[4] = 1 - player # Toggle player
                    temp_state[idx,x,y] = 0 # Piece no longer in prev location
                    if (fwd == 1 and x+fwd == 7) or \
                    (fwd == -1 and x+fwd == 0): # On King's row, king the man
                        temp_state[idx+1,x+fwd,y-1] = 1
                    else: # Not on King's row, man does not become king
                        temp_state[idx,x+fwd,y-1] = 1
                    if fwd == 1:
                        state[8,x,y] = 1 # NN layer representing BL move
                        temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                            8, x, y
                    else:
                        state[6,x,y] = 1 # NN layer representing UL move
                        temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                            6, x, y
                    legal_moves.append(temp_state)
            # Check to see if man can jump any of opponent's pieces
            jump_moves.extend(self._check_jumps(x,y,fwd,state,idx,opp_idx,board,player))
        # Get legal moves including jumps for kings
        for x, y in piece_locs[idx+1]: # Kings
            for xmove in range(-1,2,2):        
                for ymove in range(-1,2,2):
                    if -1 < x+xmove < 8 and -1 < y+ymove < 8:
                        if board[x+xmove,y+ymove] == 0: # Diag space open
                            temp_state = deepcopy(state)
                            temp_state[5:] = 0 # Erase NN layers from previous state
                            temp_state[4] = 1 - player # Toggle player
                            temp_state[idx+1,x,y] = 0 # Piece no longer in prev location
                            temp_state[idx+1,x+xmove,y+ymove] = 1
                            if xmove == 1 and ymove == 1:
                                state[9,x,y] = 1 # NN layer representing BR move
                                temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                    9, x, y
                            elif xmove == 1 and ymove == -1:
                                state[8,x,y] = 1 # NN layer representing BL move
                                temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                    8, x, y
                            elif xmove == -1 and ymove == 1:
                                state[7,x,y] = 1 # NN layer representing UR move
                                temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                    7, x, y
                            elif xmove == -1 and ymove == -1:
                                state[6,x,y] = 1 # NN layer representing UL move
                                temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                    6, x, y
                            legal_moves.append(temp_state)
            # Check to see if king can jump any of opponent's pieces
            jump_moves.extend(self._check_king_jumps(x,y,state,idx,opp_idx,board,player))
        if jump_moves: 
            state[6:10] = 0 # Clear all possible non-jump moves
            return jump_moves # Jumps are mandatory    
        return legal_moves
            
    def _check_jumps(self,x,y,fwd,state,idx,opp_idx,board,player):
        """Method intended for internal use.  Checks to see if a jump is 
        possible for a man given its position and the game state.  Function 
        recursively calls itself in case multiple jumps are possible in the 
        same turn.  All jumps are mandatory moves, so a double jump takes 
        precedence over a single jump, a triple jump over a double jump, etc.
        
        If a jump lands a man on King's row, the man is kinged and the player's
        turn is over.
        """
        jump_moves = []
        more_jumps = []
        for ydir in range(-1,2,2):
            if -1 < y+ydir < 8 and -1 < x+fwd < 8:
                if state[opp_idx,x+fwd,y+ydir] == 1 or \
                state[opp_idx+1,x+fwd,y+ydir] == 1: # Opponent's piece on diag space
                    if -1 < y+2*ydir < 8 and -1 < x+2*fwd < 8:
                        if board[x+fwd*2,y+ydir*2] == 0: # Piece is jumpable
                            temp_state = deepcopy(state)
                            temp_state[5:] = 0 # Erase NN layers from previous state
                            temp_state[idx,x,y] = 0 # Piece no longer in prev location
                            temp_state[opp_idx,x+fwd,y+ydir] = 0 # Opponent's piece jumped (if man)
                            temp_state[opp_idx+1,x+fwd,y+ydir] = 0 # Opponent's piece jumped (if king)
                            if (fwd == 1 and x+2*fwd == 7) or \
                                (fwd == -1 and x+2*fwd == 0): # On King's row, jump is over
                                temp_state[idx+1,x+2*fwd,y+2*ydir] = 1 # Man is kinged
                            else: # Check for multiple jumps
                                temp_state[idx,x+2*fwd,y+2*ydir] = 1
                                more_jumps = self._check_jumps(x+2*fwd,y+2*ydir,
                                                          fwd,temp_state,idx,
                                                          opp_idx,board,player)
                            if more_jumps:
                                #jump_moves.extend(more_jumps)
                                more_jumps = [] # Don't toggle player
                            else:
                                temp_state[4] = 1 - player # Toggle player
                            if fwd == 1 and ydir == 1: 
                                state[13,x,y] = 1 # NN layer representing BR jump
                                temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                    13, x, y
                            elif fwd == 1 and ydir == -1:
                                state[12,x,y] = 1 # NN layer representing BL jump
                                temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                    12, x, y
                            elif fwd == -1 and ydir == 1:
                                state[11,x,y] = 1 # NN layer representing UR jump
                                temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                    11, x, y
                            elif fwd == -1 and ydir == -1:
                                state[10,x,y] = 1 # NN layer representing UL jump
                                temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                    10, x, y
                            jump_moves.append(temp_state)
        return jump_moves
            
    def _check_king_jumps(self,x,y,state,idx,opp_idx,board,player):
        """Method intended for internal use.  Checks to see if a jump is 
        possible for a king given its position and the game state.  Function 
        recursively calls itself in case multiple jumps are possible in the 
        same turn.  All jumps are mandatory moves, so a double jump takes 
        precedence over a single jump, a triple jump over a double jump, etc.
        """
        jump_moves = []
        more_jumps = []
        for ydir in range(-1,2,2):
            for fwd in range(-1,2,2):
                if -1 < x+fwd < 8 and -1 < y+ydir < 8:
                    if state[opp_idx,x+fwd,y+ydir] == 1 or \
                    state[opp_idx+1,x+fwd,y+ydir] == 1: # Opponent's piece on diag space
                        if -1 < x+2*fwd < 8 and -1 < y+2*ydir < 8:
                            if board[x+fwd*2,y+ydir*2] == 0: # Piece is jumpable
                                temp_state = deepcopy(state)
                                temp_state[5:] = 0 # Erase NN layers from previous state
                                temp_state[idx+1,x,y] = 0 # Piece no longer in prev location
                                temp_state[opp_idx,x+fwd,y+ydir] = 0 # Opponent's piece jumped (if man)
                                temp_state[opp_idx+1,x+fwd,y+ydir] = 0 # Opponent's piece jumped (if king)
                                temp_state[idx+1,x+2*fwd,y+2*ydir] = 1  # Move piece to new location
                                more_jumps = self._check_king_jumps(x+2*fwd,y+2*ydir,
                                                          temp_state,idx,
                                                          opp_idx,board,player)
                                if more_jumps:
                                    #jump_moves.extend(more_jumps)
                                    more_jumps = [] # Don't toggle player
                                else:
                                    temp_state[4] = 1 - player # Toggle player
                                if fwd == 1 and ydir == 1: 
                                    state[13,x,y] = 1 # NN layer representing BR jump
                                    temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                        13, x, y
                                elif fwd == 1 and ydir == -1:
                                    state[12,x,y] = 1 # NN layer representing BL jump
                                    temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                        12, x, y
                                elif fwd == -1 and ydir == 1:
                                    state[11,x,y] = 1 # NN layer representing UR jump
                                    temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                        11, x, y
                                elif fwd == -1 and ydir == -1:
                                    state[10,x,y] = 1 # NN layer representing UL jump
                                    temp_state[14,0,0], temp_state[14,0,1], temp_state[14,0,2] = \
                                        10, x, y
                                jump_moves.append(temp_state)
        return jump_moves
    
    def determine_outcome(self, history, legal_moves=[]):
        """Given game history as an input, determine if the game is over.  
        If the game is over, determine the winner (or if the game is a draw).
        
        The win condition is to either capture all of your opponent's pieces or
        to make the last move - meaning that the opponent is unable to move 
        their pieces due to being blocked.
        
        According to the WCDF, a game shall be declared a draw when both of
        the following conditions are true:
        
        1. Neither player has advanced an uncrowned man towards the king-row 
           during their own previous 40 moves.
        2. No pieces have been removed from the board during their own previous 
           40 moves.
        
        I interpret this to mean that the conditions should be checked over the 
        last 80 moves of the game (last 40 moves for each player).
        """
        state = history[-1]
        current_num_pieces = np.sum(state[0:4]) # Number of pieces on board
        current_player = int(state[4,0,0])
        last_player_to_move = 1 - current_player
        if not legal_moves:
            legal_moves = self._check_moves(history) # Can player make a move?
        man_moved, piece_jumped = True, True # Default if <80 moves played
        if len(history) >= 80:
            man_moved = False # Draw condition #1
            piece_jumped = False # Draw condition #2
            for cnt, move in enumerate(reversed(history[-80:])): # Check for draw conditions
                if np.sum(move[0:4]) != current_num_pieces: 
                    piece_jumped = True
                    state[5] = cnt / 80 # NN layer representing draw counter
                    break
                if not ((move[0] == state[0]).all() and (move[2] == state[2]).all()):
                    man_moved = True
                    state[5] = cnt / 80 # NN layer representing draw counter
                    break
        if np.sum(state[2:4]) == 0: # All player 2 pieces jumped
            done = True
            outcome = 'player1_wins'
        elif np.sum(state[0:2]) == 0: # All player 1 pieces jumped
            done = True
            outcome = 'player2_wins'
        elif not legal_moves: # No legal moves for player
            if last_player_to_move == 0:
                done = True
                outcome = 'player1_wins'
            else:
                done = True
                outcome = 'player2_wins'
        elif (not man_moved) and (not piece_jumped):
            done = True
            outcome = 'draw'
            state[5] = 1 # NN layer representing draw counter
        else: 
            done = False # Game is still in progress
            outcome = None
        return done, outcome
        
    def print_board(self):
        """Print a visual representation of the current game state to the 
        console."""
        player = int(self.state[4,0,0])
        player_mark = self.player1_man if player == 0 else self.player2_man 
        board_pieces = (self.state[0] - self.state[2]) + \
                        2*(self.state[1] - self.state[3])
        board_table = []
        for row_idx, row in enumerate(board_pieces):
            row_list = []
            for col_idx, square in enumerate(row):
                if square == 1: row_list.append(self.player1_man)
                if square == -1: row_list.append(self.player2_man)
                if square == 2: row_list.append(self.player1_king)
                if square == -2: row_list.append(self.player2_king)
                if square == 0: 
                    if (not (row_idx % 2) and not (col_idx % 2)):
                        row_list.append('.') # Unused dark squares    
                    elif ((row_idx % 2) and (col_idx % 2)):
                        row_list.append('.') # Unused dark squares    
                    else:
                        row_list.append('') # Empty squares
            board_table.append(row_list)
        print(tabulate(board_table, tablefmt='fancy_grid'))
        if not self.done:
            print('Move #{}: It\'s now Player {}\'s turn ({})'
                  .format(self.move_count+1, player+1, player_mark))
        else:
            print('Game over after {} moves! The outcome is: {}'
                  .format(self.move_count+1, self.outcome))
                
    def current_player(self, state):
        """Return which player's turn it is for a given input state."""
        player = int(state[4,0,0])
        if player == 0:
            return 'player1'
        else:
            return 'player2'
    
    def reset(self):
        """Reset the board so it's ready for the next game."""
        self.state = np.zeros((15,8,8), dtype=float)
        self.init_board()
        self.history = [self.state]
        self.legal_next_states = self.get_legal_next_states(self.history)
        self.move_count = 0
        self.done = False
        self.outcome = None
    
    def init_board(self):
        """Place Checkers pieces on board in their starting positions."""
        for row in range(0,8):
            for col in range(0,8):
                if row % 2 != col % 2: # Row and column not both odd or even
                    if row < 3:
                        self.state[0,row,col] = 1
                    elif row > 4:
                        self.state[2,row,col] = 1
                 
    def predict(self, state):
        """Use the supplied neural network to predict the Q-value of a given
        state, as well as the prior probabilities of its child states.
        Masks any non-valid probabilities, and re-normalizes the remaining
        probabilities to sum to 1.
        """
        prob_vector, q_value = self.neural_net.predict(state[:14].reshape(1,14,8,8))
        action_mask = self.create_action_mask(state)
        prob_vector *= action_mask
        prob_vector = prob_vector / np.sum(prob_vector)
        return prob_vector[0], q_value[0][0]

    def create_action_mask(self, state):
        """Creates mask to apply to neural network predictions.  Masks all 
        invalid actions predicted by the neural network.
        """
        action_array = state[6:14]
        actions = action_array.flatten()
        valid_squares = np.ma.array(actions, mask=32*[1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1])
        action_mask = valid_squares[valid_squares.mask == False]
        return action_mask
                    
    def set_prior_probs(self, child_nodes, prob_vector):
        """Takes as input a list of the parent node's child nodes, and the
        probability vector generated by running the parent's state through the
        neural network.  Assigns each child node its corresponding prior
        probability as predicted by the neural network.
        """
        for child in child_nodes:
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
            child._prior_prob = prob_vector[int(idx)]


class Checkers_GUI:
    """A class to display a Pygame representation of the Checkers board."""
    def __init__(self, game_env):
        """Initializes a Pygame GUI to visualize the game of Checkers."""
        # Reference to game environment
        self.game_env = game_env
        # Initialize Pygame display
        pygame.init()
        pygame.mixer.quit() # Fixes bug with high PyGame CPU usage
        self.game_width = 600
        self.game_height = 600
        self.gameDisplay = pygame.display.set_mode((self.game_width, 
                                                    self.game_height))
        pygame.display.set_caption('Checkers')
        # Define GUI variables
        self.sq_dim = 50 # 50x50 pixel squares
        self.board_width = 8*self.sq_dim # Pixels
        self.board_height = 8*self.sq_dim # Pixels
        self.board_offset = 100 # Pixels
        self.move_delay = 1 # Seconds between animations
        self.GREEN = (1,50,32)
        self.BROWN = (101,67,33)
        self.WHITE = (255,255,255)
        self.gridfont = pygame.font.SysFont('Segoe UI', 32)
        self.statusfont = pygame.font.SysFont('Segoe UI', 28)
        # Load images
        self.board = pygame.image.load('img/board.png').convert_alpha()
        self.red_checker = pygame.image.load('img/red_checker.png').convert_alpha()
        self.black_checker = pygame.image.load('img/black_checker.png').convert_alpha()
        self.red_king = pygame.image.load('img/red_king.png').convert_alpha()
        self.black_king = pygame.image.load('img/black_king.png').convert_alpha()
        self.ghost_checker = pygame.image.load('img/ghost_checker.png').convert_alpha()
        self.ghost_king = pygame.image.load('img/ghost_king.png').convert_alpha()
        self.move_checker = pygame.image.load('img/move_checker.png').convert_alpha()
        self.move_king = pygame.image.load('img/move_king.png').convert_alpha()
        self.select_sq = pygame.image.load('img/select_sq.png').convert_alpha() 
        self.move_sq = pygame.image.load('img/move_sq.png').convert_alpha()
        # Draw display
        self._set_board(game_env.state)
        
    def _set_board(self, state):
        """Blit Checkers pieces on GUI board in their starting positions."""
        # Get current game state information
        state = self.game_env.state
        move_count = self.game_env.move_count+1
        player = player = int(state[4,0,0])
        player1_man = 'red'
        player2_man = 'black'
        player_mark = player1_man if player == 0 else player2_man 
        done = self.game_env.done
        outcome = self.game_env.outcome
        # Blit green background
        pygame.draw.rect(self.gameDisplay, self.GREEN, (0,0,
                                                        self.game_width,
                                                        self.game_height)) 
        # Blit game board
        self.gameDisplay.blit(self.board, 
                              (self.board_offset,self.board_offset), 
                              (0, 0, self.board_width, self.board_height))
        # Blit grid numbers for each row and column
        for num in range(8):
            grid_text = self.gridfont.render(str(num+1), True, self.WHITE)
            self.gameDisplay.blit(grid_text, 
                                  (self.board_offset+self.sq_dim//2-5+num*self.sq_dim,
                                   self.board_offset-30))
            self.gameDisplay.blit(grid_text, 
                                  (self.board_offset-30,
                                   self.board_offset+self.sq_dim//2-10+num*self.sq_dim))
        # Blit status message
        if not done:
            status_str = 'Move #{}: It\'s now Player {}\'s turn ({})'.format(move_count, player+1, player_mark)
        else:
            status_str = 'Game over after {} moves! The outcome is: {}'.format(move_count, outcome)
        status_text = self.statusfont.render(status_str, True, self.WHITE)
        self.gameDisplay.blit(status_text, 
                              (self.board_offset,
                               self.board_offset+self.board_height+30))
        # Blit Checkers pieces
        for row in range(0,8):
            for col in range(0,8):
                if row % 2 != col % 2: # Row and column not both odd or even
                    if row < 3:
                        self.gameDisplay.blit(self.red_checker,
                                          (self.board_offset+self.sq_dim*col, 
                                           self.board_offset+self.sq_dim*row))
                    elif row > 4:
                        self.gameDisplay.blit(self.black_checker,
                                          (self.board_offset+self.sq_dim*col, 
                                           self.board_offset+self.sq_dim*row))
        self._blit_possible_moves(state)
        self.update_screen()

    def _blit_possible_moves(self, state):
        """Blits a visual indicator of which pieces can make a move, and the 
        locations that the pieces can move to.
        """
        player = int(state[4,0,0])
        player_idx = player * 2
        possible_moves = np.sum(state[6:14], axis=0)
        poss_x, poss_y = np.where(possible_moves > 0)
        for x, y in zip(poss_x, poss_y): # Highlight pieces that can move
            self.gameDisplay.blit(self.select_sq,
                                      (self.board_offset+self.sq_dim*y, 
                                       self.board_offset+self.sq_dim*x)) 
        # Place ghost pieces to indicate which squares pieces can move to
        shift = [(-1,-1), (-1,1), (1,-1), (1,1), (-2,-2), (-2,2), (2,-2), (2,2)]
        for idx in range(6,14): 
            move_x, move_y = np.where(state[idx] > 0)
            for x, y in zip(move_x, move_y):
                if state[player_idx, x, y] == 1:
                    ghost = self.ghost_checker
                else:
                    ghost = self.ghost_king
                self.gameDisplay.blit(ghost,
                  (self.board_offset+self.sq_dim*(y + shift[idx-6][1]), 
                   self.board_offset+self.sq_dim*(x + shift[idx-6][0])))                     
            
    def update_screen(self):
        """Update Pygame display after blitting operations."""
        pygame.display.update()

    def render(self):
        """Renders the new board state every time the step() method is called.
        
        Blits an indicator of the move selected in the previous state, updates
        the screen, pauses for a moment, and then redraws the game board with
        the current state's piece positions and possible moves.
        """
        # Get current game state information
        prev_state = self.game_env.history[-2]
        state = self.game_env.state
        move_count = self.game_env.move_count+1
        player = player = int(state[4,0,0])
        player1_man = 'red'
        player2_man = 'black'
        player_mark = player1_man if player == 0 else player2_man 
        done = self.game_env.done
        outcome = self.game_env.outcome
        # Blit selection animation
        old_xy, new_xy = self._states_to_piece_positions(prev_state, state)
        self._blit_selected_move(prev_state, state, old_xy, new_xy)
        self.update_screen()
        time.sleep(self.move_delay)
        # Blit board and pieces
        self.gameDisplay.blit(self.board, 
                              (self.board_offset,self.board_offset), 
                              (0, 0, self.board_width, self.board_height))
        self._blit_pieces(state)
        self.update_screen()
        time.sleep(self.move_delay)        
        # Erase old status text and blit new status text
        pygame.draw.rect(self.gameDisplay, self.GREEN, (0,
                        self.board_height+self.board_offset,
                        self.game_width,
                        self.game_height-self.board_height-self.board_offset)) 
        if not done:
            status_str = 'Move #{}: It\'s now Player {}\'s turn ({})'.format(move_count, player+1, player_mark)
            status_text = self.statusfont.render(status_str, True, self.WHITE)
            self.gameDisplay.blit(status_text, (self.board_offset,
                               self.board_offset+self.board_height+30))
        else:
            status_str = 'Game over after {} moves!'.format(move_count)
            status_text = self.statusfont.render(status_str, True, self.WHITE)
            self.gameDisplay.blit(status_text, (self.board_offset,
                               self.board_offset+self.board_height+30))
            status_str = 'The outcome is: {}'.format(outcome)
            status_text = self.statusfont.render(status_str, True, self.WHITE)
            self.gameDisplay.blit(status_text, (self.board_offset,
                               self.board_offset+self.board_height+62))
        
        # Blit new possible moves
        self._blit_possible_moves(state)
        self.update_screen()
        
    def _states_to_piece_positions(self, prev_state, state):
        """Given a previous state and the current state, return two (x,y)
        coordinates.  The first coordinate will be the location of the piece
        that was moved, and the second coordinate will be the location that 
        the piece moved to.
        """
        board = prev_state[0] + 2*prev_state[1] + 3*prev_state[2] + 4*prev_state[3]
        nboard = state[0] + 2*state[1] + 3*state[2] + 4*state[3]
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
        return (xold,yold), (xnew,ynew)

    def _blit_selected_move(self, prev_state, state, old_xy, new_xy):
        """Once player has selected a move, this function blits an indication 
        of which piece will be moved and the square it will be moved to.
        """
        move_sq, sq_dim = self.move_sq, self.sq_dim
        player = int(prev_state[4,0,0])
        player_idx = player * 2
        if state[player_idx, new_xy[0], new_xy[1]] == 1:
            new_piece = self.move_checker
        else:
            new_piece = self.move_king
        self.gameDisplay.blit(move_sq, (self.board_offset+sq_dim*old_xy[1], 
                                          self.board_offset+sq_dim*old_xy[0]))
        self.gameDisplay.blit(new_piece, (self.board_offset+sq_dim*new_xy[1], 
                                          self.board_offset+sq_dim*new_xy[0])) 

    def _blit_pieces(self, state):
        """Iterates through the first four layers of the state array and blits
        the corresponding game pieces to the board.
        """
        piece = [self.red_checker, self.red_king, 
                 self.black_checker, self.black_king]
        for idx in range(4):
            xloc, yloc = np.where(state[idx] == 1)
            for x, y in zip(xloc, yloc):
                self.gameDisplay.blit(piece[idx],
                          (self.board_offset+self.sq_dim*y, 
                           self.board_offset+self.sq_dim*x)) 
                
    def _create_board_image(self):
        """Assemble Checkers board into a single image and save to disk.
        Only needs to be run when a change to the Checkers board design is 
        desired.
        """
        board = pygame.Surface([self.board_width, self.board_height])
        wood = pygame.image.load('img/wood5.png').convert()
        black_sq = pygame.Surface((self.sq_dim, self.sq_dim))
        black_sq.set_alpha(160)
        black_sq.fill(self.BROWN)
        board.blit(wood,(0,0), (0, 0, self.board_width, self.board_height))        
        for row in range(8):
            for col in range(8):
                if row % 2 != col % 2:
                    board.blit(black_sq, (self.sq_dim*col, self.sq_dim*row))
        pygame.image.save(board, 'img/board.png')
        
    def close_gui(self):
        """Close Pygame GUI."""
        pygame.quit()


def test_game():
    """Test function to validate functionality of Checkers class.
    Plays a game of Checkers by randomly selecting moves for both players.
    """
    GUI = True
    checker = Checkers()
    checker.print_board()
    if GUI: checker_gui = Checkers_GUI(checker)
    input('Press enter to continue...')
    while not checker.done:
        legal_next_states = checker.get_legal_next_states(checker.history)
        move_idx = np.random.randint(0,len(legal_next_states))
        checker.step(legal_next_states[move_idx])
        checker.print_board()
        if GUI: checker_gui.render()
        input('Press enter to continue...')
        if checker.done and GUI: checker_gui.close_gui()
    
    
if __name__ == '__main__':
    test_game()