#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# MCTS.py
#
# Revision:     1.00
# Date:         11/07/2020
# Author:       Alex
#
# Purpose:      Contains classes to implement the Monte Carlo Tree Search  
#               algorithm for two-player turn-based games.
#
# Classes:
# 1. MCTS       -- Class to implement the MCTS algorithm and interact with the
#                  game environment.
# 2. MCTS_Node  -- A class representing a MCTS node.   
#
# Notes:
# 1. Terminology based on "A Survey of Monte Carlo Tree Search Methods" by
#    Browne et al.
#
###############################################################################
"""

import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta
import multiprocessing as mp


MAX_PROCESSORS = mp.cpu_count()


class MCTS:
    """Class to implement the Monte Carlo Tree Search algorithm.
    All methods are implemented as class methods to avoid the need to pass a 
    MCTS class instance to every node in the tree.
    """
    @classmethod
    def __init__(cls, **kwargs):
        """Initialize MCTS parameters and pass in the game environment."""
        cls.game_env = kwargs['GAME_ENV']
        cls.uct_c = kwargs['UCT_C'] 
        cls.constraint = kwargs['CONSTRAINT'] # Either 'time' or 'rollout'
        cls.budget = kwargs['BUDGET'] # In seconds or max rollouts
        cls.multiproc = kwargs['MULTIPROC'] # Uses multiprocessing if True
        cls.neural_net = kwargs['NEURAL_NET'] # If True NN is used for rollouts
        cls.verbose = kwargs['VERBOSE'] # Suppress printed messages if False
        cls.training = kwargs['TRAINING'] # False for competitive play
        cls.alpha = kwargs['DIRICHLET_ALPHA']
        cls.epsilon = kwargs['DIRICHLET_EPSILON']
        cls.tau = kwargs['TEMPERATURE_TAU']
        cls.tau_decay = kwargs['TEMPERATURE_DECAY']
        cls.tau_decay_delay = kwargs['TEMP_DECAY_DELAY']
        if cls.multiproc and not cls.neural_net: 
            cls.pool = mp.Pool(MAX_PROCESSORS)
        
    @classmethod 
    def tree_policy(cls, node):
        """Defines policy for selecting and expanding nodes in MCTS tree.
        
        Must visit any unvisited children first to add them to the tree.  
        A simulation is then performed from the newly added child node.  If all 
        of the node's children have already been visited at least once 
        (node fully expanded), continue moving down the tree according to the 
        best child criterion.
        """
        if node.unvisited_child_states:
            if cls.neural_net:
                prob_vector, q_value = cls.game_env.predict(node.state)
                for _ in range(len(node.unvisited_child_states)):
                    next_state = node.unvisited_child_states.pop()
                    child_node = MCTS_Node(next_state, parent=node)
                    node.children.append(child_node) # Expand tree
                cls.game_env.set_prior_probs(node.children, prob_vector)
                node.backpropagation(q_value, node.player) 
            else:
                next_state = node.unvisited_child_states.pop()
                child_node = MCTS_Node(next_state, parent=node)
                node.children.append(child_node) # Expand tree
                if cls.computational_budget():
                    if cls.multiproc:
                        outcomes = cls.pool.map(cls.default_policy, 
                                                MAX_PROCESSORS*[child_node])
                        for outcome in outcomes:
                            child_node.backpropagation(outcome, child_node.player)
                    else:
                        child_node.simulation() # Begin simulation phase
        else:
            if not node.terminal:
                child_node = cls.select_child(node) 
                if child_node.terminal: # Child is terminal state
                    child_node.simulation() # Begin simulation phase
                else:
                    child_node.selection() # Continue selection phase
            else: # Node is terminal state
                outcome = cls.determine_outcome(node)
                node.backpropagation(outcome, node.player)               
    
    @classmethod
    def select_child(cls, node):
        """Descend through the tree according to maximum UCT value."""
        if cls.neural_net:
            num_nodes = len(node.children)
            prior_probs = np.array([child.p for child in node.children])
            Psa_probs = (1-cls.epsilon)*prior_probs + \
                cls.epsilon * np.random.dirichlet([cls.alpha]*num_nodes)
            uct_values = [child.q + cls.uct_c * Psa  * 
                          (node.n ** 0.5) / (1 + child.n) 
                          for child, Psa in zip(node.children, Psa_probs)]
        else:
            uct_values = [child.q + 2.0 * cls.uct_c *
                          (2.0 * np.log(node.n) / child.n) ** 0.5
                          for child in node.children]
        return node.children[np.argmax(uct_values)]
    
    @classmethod
    def default_policy(cls, node):
        """Simulate the game to produce a value estimate according to the 
        desired policy.  
        
        Standard MCTS plays randomly until a terminal state is reached, and the 
        resulting reward is backpropagated through the tree.  Alternatively, a 
        policy network can determine which moves will be selected for 
        simulation, and an evaluation function can estimate the value of a 
        non-terminal state.
        
        The RNG is re-seeded if multiprocessing is enabled so that the outcomes 
        of the processes aren't identical.
        """
        if not cls.neural_net:
            if cls.multiproc: np.random.seed() # Ensure different RNG seeds
            game_sim = deepcopy(cls.game_env) # Copy of environment for simulation
            # Play states that occurred between the root node and the current node
            for move_num, state in enumerate(node.history):
                if (move_num) > game_sim.move_count: game_sim.step(state)
            # Simulate from current node state to terminal state
            while not game_sim.done: 
                legal_next_states = game_sim.legal_next_states
                move_idx = np.random.randint(0,len(legal_next_states))
                game_sim.step(legal_next_states[move_idx])
            return game_sim.outcome, node.player
        else:
            done, outcome = cls.determine_outcome(node) # Terminal state
            return outcome, node.player
        
    @classmethod
    def determine_reward(cls, player, outcome, parent_node, sim_player):
        """Determine the node's reward based on the node's player and the
        outcome of the game.
        
        If the player of the current state is player 1, that means that the 
        current node is a child node (potential move) of player 2.  Therefore 
        the rewards are flipped: a player 1 victory on a player 1 node receives 
        a reward of -1.  A player 2 victory on a player 1 node receives a 
        reward of +1.  This adjusts the UCT value of the node from the 
        opponent's perspective, and incentivizes the MCTS to choose nodes that 
        favor player 2 when representing it during the selection phase.  The 
        same is true for player 2 nodes so that they are chosen from player 1's 
        perspective during the selection process.
        
        The exception is if the game allows multiple moves per turn, e.g. a 
        multi-jump in Checkers.  To handle this case, the parent node's player
        must be checked against the outcome.
        """
        if parent_node is not None:
            parent_player = parent_node.player
        else: # Root node
            try:
                parent_player = cls.current_player(cls.game_env.history[-2])
            except:
                parent_player = 'player1' if player == 'player2' else 'player2'
        if type(outcome) == str: # Random rollout
            if outcome == 'player1_wins':
                reward = 1 if parent_player == 'player1' else -1
            elif outcome == 'player2_wins':
                reward = 1 if parent_player == 'player2' else -1
            elif outcome == 'draw':
                reward = 0
            return reward
        else: # Outcome is state's estimated Q-value from neural network
            if sim_player != parent_player:
                return -1*outcome # Reverse estimated Q-value
            else:
                return outcome
        
    @classmethod
    def computational_budget(cls):
        """Check computational constraint on the MCTS: e.g. time, memory, or 
        iterations. Return False if computational budget is exceeded.
        """
        if cls.constraint == 'time':
            if datetime.now() >= cls.start_time + timedelta(seconds=cls.budget):
                return False
        elif cls.constraint == 'rollout':
            if cls.rollout_count >= cls.budget:
                return False
        else:
            raise ValueError('Invalid MCTS computational constraint!')
        return True

    @classmethod        
    def get_legal_next_states(cls, history):
        """Query the game environment to get the legal next states for a 
        given state of the game.
        """
        return cls.game_env.get_legal_next_states(history)
            
    @classmethod 
    def begin_tree_search(cls, root_node):
        """Begin the Monte Carlo Tree Search by calling the root node's 
        selection method.  The search will continue to expand the tree until
        the computational budget is exhausted.
        """
        cls.start_time = datetime.now()
        cls.rollout_count = 0
        if cls.verbose: print('Starting search!')
        while cls.computational_budget():
            root_node.selection()
        if cls.verbose: 
            print('Stopped  search after {} rollouts and {} duration!'
              .format(cls.rollout_count, 
                      str(datetime.now()-cls.start_time)[2:-4]))
                
    @classmethod 
    def best_child(cls, node, criterion='robust'):
        """After search is terminated, select the winning action based on 
        desired selection criterion.
        """
        if cls.neural_net: criterion = 'robust'
        if criterion == 'max': # Max child: child with highest reward
            rewards = [child.w for child in node.children]
            return node.children(np.argmax(rewards))
        elif criterion == 'robust': # Robust child: most visited child
            visits = [child.n for child in node.children]
            if not cls.training or cls.tau <= 0:
                return node.children[np.argmax(visits)]
            else:
                expon_visits = [n ** (1/cls.tau) for n in visits]
                total = np.sum(expon_visits)
                probs = [n / total for n in expon_visits]
                if cls.game_env.move_count > cls.tau_decay_delay:
                    cls.tau -= cls.tau_decay
                    if np.isclose(cls.tau, 0): cls.tau = 0
                return np.random.choice(node.children, p=probs)
        else:
            raise ValueError('Invalid winner selection criterion!')
            
    @classmethod
    def new_root_node(cls, old_root):
        """Return a new root node to run the Monte Carlo Tree Search from.
        
        Once the opponent has moved, the previous root node should be 
        replaced by the node representing the opponent's action.  This prunes 
        the tree of all of the possible actions that the opponent didn't take, 
        while preserving the simulation results of the node representing
        the action that the opponent did take.  
        
        If this node exists in the tree, then return it as the new root node.  
        Otherwise, the opponent's move did not get visited during the previous 
        turn's selection phase and must be created here and returned as the new 
        root node.
        
        In some games it's possible that a player will make multiple moves in
        one turn (such as a double jump in Checkers).  A WHILE loop checks to
        see if the other player made multiple consecutive moves, and if so the 
        following nested FOR loops will traverse through the search tree until
        the node is found that corresponds to the current state of the game.
        """
        new_state = cls.game_env.state
        counter = 1 
        state_idx = -3
        while True: # Check if a player made consecutive moves
            if cls.game_env.current_player(cls.game_env.history[-2]) != \
                cls.game_env.current_player(cls.game_env.history[state_idx]):    
                break
            counter += 1
            state_idx -= 1
        new_root = old_root        
        for idx in range(-counter,0,1):
            for child in new_root.children:
                if (child.state == cls.game_env.history[idx]).all():
                    new_root = child
                    break
        if (new_root.state == new_state).all():
            new_root.parent = None
            return new_root
        else: # This possible move didn't get visited during search
            new_root = MCTS_Node(new_state)
            new_root.history = deepcopy(cls.game_env.history)
            raise ValueError('All child nodes should be visited!  Consider '
                              'increasing number of rollouts or comment out this'
                              'error.')
            return new_root
    
    @classmethod
    def determine_outcome(cls, node):
        """Query the game environment to determine the winner (if any) of the
        game.  
        """
        return cls.game_env.determine_outcome(node.history)

    @classmethod
    def current_player(cls, state):
        """Query the game environment to determine which player's turn it is
        for the given state.
        """
        return cls.game_env.current_player(state)
    
    @classmethod
    def print_tree(cls, root_node, max_tree_depth=10):
        """Create a copy of the node and recursively traverse through it to 
        print a tree diagram.  Prints total value and visits of each node in 
        (Q/N) format.  Explores the tree in a depth-first search (DFS) manner.
        """
        root_depth = root_node.depth
        node = deepcopy(root_node)
        cls.traverse_tree(node, max_tree_depth, root_depth)
    
    @classmethod
    def traverse_tree(cls, node, max_tree_depth, root_depth):
        """Recursively moves through the tree until it has either printed 
        every node in the tree or it has met the maximum depth limit.
        
        If the node has children, always chooses the last child in the list of 
        children to visit.  If the node has no children (or is at the depth 
        limit), uses the pop() function to remove the last child from the 
        parent node's list.  Then moves to the parent node to check if it has 
        any other children to visit.  
        """
        if not node.printed:
            node_w_str = "{0}".format(str(round(node.w, 1) if node.w % 1 else int(node.w)))
            print('\t'*(node.depth-root_depth) + '|- ({}/{}) ({:.1f}%)'
                  .format(node_w_str, node.n, node.pwin))
            node.printed = True
        if node.children: # Choose last child in list
            if (node.children[-1].depth - root_depth) <= max_tree_depth:
                cls.traverse_tree(node.children[-1], max_tree_depth, root_depth)
        if node.parent: # Reached max depth of the tree, move back up
            if node.parent.children: node.parent.children.pop()
            cls.traverse_tree(node.parent, max_tree_depth, root_depth)


class MCTS_Node:
    """A class representing a MCTS node.  The node stores its total reward, 
    its number of visits, its parent node, its child nodes, and a history of 
    all prior states.  All decision making is offloaded to the MCTS class.
    """
    def __init__(self, state, parent=None, initial_state=None):
        """Initialize node's state, parent node, child nodes, and a history of
        all prior states.  Initialize total reward and number of visits to 
        zero, and check to see if node is a terminal node.  
        
        If the MCTS root node is player 2, initialize its first instance with 
        the initial state of the game so that player 2's history is 
        synchronized with the game environment's move count.  This only has to 
        be done once at the start of the game if the MCTS is player 2.
        """
        self.state = state
        self.player = MCTS.current_player(self.state)
        self.parent = parent
        if parent:
            self.history = parent.history.copy()
            self.history.append(state)
        else:
            self.history = [state]
            if initial_state is not None: self.history.insert(0, initial_state)
        self.depth = len(self.history)
        self.children = []
        self._number_of_visits = 0
        self._total_reward = 0
        self._prior_prob = 0
        self.unvisited_child_states = MCTS.get_legal_next_states(self.history)
        self.terminal = False if self.unvisited_child_states else True
        self.printed = False # Used by MCTS.print_tree()
            
    @property
    def w(self):
        """Property decorator used to return total reward value of node."""
        return self._total_reward

    @property
    def n(self):
        """Property decorator used to return number of node visits."""
        return self._number_of_visits    
    
    @property
    def q(self):
        """Property decorator used to return mean reward value of node."""
        try:
            return self.w / self.n
        except ZeroDivisionError:
            return 0
    
    @property
    def p(self):
        """Property decorator used to return prior probability of node."""
        return self._prior_prob
    
    @property
    def pwin(self):
        """Property decorator used to return NN's confidence of winning."""
        return np.round((self.q+1)/2*100, 1)
    
    def selection(self):
        """The selection phase of MCTS.  Choose an action based on the MCTS
        tree policy.
        """
        MCTS.tree_policy(self)
           
    def simulation(self):
        """Simulate the game from the current node according to the default 
        policy.  Backpropagate the outcome back to the root node.
        """
        outcome, sim_player = MCTS.default_policy(self)
        self.backpropagation(outcome, sim_player)
    
    def backpropagation(self, outcome, sim_player):
        """Simulation result is backpropagated through the selected nodes
        to the root node, and their statistics are updated.
        """
        reward = MCTS.determine_reward(self.player, outcome, self.parent, 
                                       sim_player)
        self._number_of_visits += 1
        self._total_reward += reward
        if self.parent:
            self.parent.backpropagation(outcome, sim_player)
        else:
            MCTS.rollout_count += 1       
