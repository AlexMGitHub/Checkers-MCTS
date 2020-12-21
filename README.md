# Training a Neural Network to play Checkers with Monte Carlo Tree Search

![Agent playing Checkers](docs/images/Minesweeper_Agent_Playing.gif "Checkers self-play!")

## Overview
I trained a neural network to play Checkers through self-play using Monte Carlo Tree Search.  The agent achieves **TBD** after **TBD** training iterations.  This required about **TBD** days of training on a laptop with an Intel Core i7-6820HQ CPU @ 2.70GHz and an NVIDIA Quadro M2002M GPU (CUDA Compute Capability 5.0).

I wrote the code in Python 3.7 and used Keras 2.4.3 (GPU-enabled with Tensorflow backend), PyGame 1.9.6[\*](#footnotes), and Matplotlib 3.3.2.  A **requirements.txt** file is included for convenience.  

I have attempted to replicate the approach DeepMind took when creating [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go), albeit with a fraction of the computational resources.  I have written several Python classes to implement the Checkers environment, the MCTS algorithm, and DeepMind's training pipeline.  The Python files of interest are:

 * **Checkers** is the agent-agnostic environment which contains all of the functions related to the game of Checkers, including a PyGame GUI. 
 * **MCTS** Contains classes to implement the Monte Carlo Tree Search algorithm for two-player turn-based games.
 * **train_Checkers** Initiates the DeepMind training pipeline of training data generation through self-play, neural network training, and network evaluation.
 * **play_Checkers** imports a saved neural network model and uses it to play a demonstration game of Checkers using the Monte Carlo Tree Search algorithm.
 * **training_pipeline** Contains classes to generate Checkers training data and to create a tournament to compare the performance of two different trained neural networks.

The following sections provide an overview of DeepMind's approach to reinforcement learning and how I adapted the techniques used in *AlphaZero* for the game of Checkers.

## DeepMind's Approach to Deep Reinforcement Learning
[I previously implemented](https://github.com/AlexMGitHub/Minesweeper-DDQN) an agent that learned to play Minesweeper based on DeepMind's work with Atari games.  DeepMind published several papers between 2013 and 2016 that used deep double Q-learning to train neural networks to play a variety of Atari games at better-than-human performance.  That's not to say *super*-human performance, but the agent often achieved a higher score than the human player used as a baseline for the game.  I found that my Minesweeper agent won 23.6% of its games on Expert difficulty.  This is a far higher win rate than what I personally could achieve as a human player, but a "perfect" player would be expected to win more than 35% of its games.  Although successful in many of the Atari games, in some games DeepMind's trained agents performed significantly worse than the human baseline.  This earlier mixed success on Atari games was blown out of the water by the 4-1 victory of [AlphaGo against Lee Sedol](https://deepmind.com/research/case-studies/alphago-the-story-so-far) in March 2016.  


### AlphaGo, AlphaGo Zero, and AlphaZero

Atari games are single player, Go/Shogi/Chess are two player (self-play).
Chose Checkers as a simpler game that may be possible to get good performance with far less computation, but still complicated enough not to be trivial (like Tic-Tac-Toe).
branching factor requires nns to narrow down search
## Monte Carlo Tree Search

### Overview
I (and most blogs I found) referenced [[5]](#references) for a fantastic overview of the Monte Carlo Tree Search algorithm.  The algorithm creates a tree where each node represents a possible move or game state, and stores statistics about the desirability of that potential move.  The root node represents the current game state, and its child nodes are the legal possible moves that can be made from that root node.  Each child node has children of its own, which are possible moves that the opponent can make.  These "grandchild" nodes have children of their own, which are moves that the original player can make after his opponent has moved, and so on.  The tree quickly grows large even for relatively simple games, and each branch eventually ends in a terminal node that represents one possible game outcome (win, loss, or draw).  The [branching factor](https://en.wikipedia.org/wiki/Game_complexity#Complexities_of_some_well-known_games) of games like Chess and Go are so large that it's not possible to represent every potential move in the tree, and so the tree search is constrained by a user-defined computational budget.

<p align="center">
<img src="docs/images/mcts_phases.png" title="MCTS Diagram" alt="MCTS Diagram" width="640"/>
</p>

Given these computational restraints, the tree search must proceed in such a way as to balance the classic trade-off between exploration and exploitation.  The above image is taken from [[5]](#references) and gives a visual overview of the four phases of the Monte Carlo Tree Search algorithm.  In each iteration of the MCTS algorithm, a *tree policy* decides which nodes to select as it descends through the tree.  Eventually, the tree policy will select a node that has children that have not yet been added to the tree (expandable nodes).  This new node is added to the tree, and a simulation or "rollout" is played out according to a *default policy*.  The simulation starts at the newly added node and continues until it reaches a terminal state.  Finally, the outcome of the simulation is backpropagated up the tree to the root node, and the statistics of each node along the way are updated.  This is a simple algorithm, but what makes it really effective are the two policies: the tree policy and default policy. 

### Tree Policy
The tree policy balances the exploration/exploitation trade-off.  It does this by treating the choice of child node as a multi-armed bandit problem, and uses a variant of the Upper Confidence Bound algorithm called UCT (Upper Confidence bounds applied to Trees).  As mentioned in the previous section, each node stores statistics that are updated whenever a simulation outcome is backpropagated through it.  The two statistics used by UCT are:
1. q<sub>j</sub> The total reward or Q-value of the node
2. n<sub>j</sub> The number of visits to the node



### Default Policy


2 policies
uct
best child typically most visited child (robust)
deepmind modified algorithm, discuss more in following sections

### Implementation
How does it work with 2 players?

### Reward Structure
Alternating reward structure for each node - reverse of the current player as it is a child node of the other player.

### Parallel Processing

### Validation with Tic-Tac-Toe


## Checkers Mechanics

## The DeepMind Training Pipeline

### Adding Stochasticity
Neural networks are deterministic, will play the same moves in the same situation.

Add Dirichlet noise to neural network prior probabilities prediction.  (Show UCT formula).  Means that MCTS will tend to play out differently for the same state, which means that games between the same neural network are no longer deterministic and will tend to differ.  Show equation with sigma and U(s,a).  Discuss value for C - didn't bother using variable C, but why choose particular constant?  Also discuss alpha value for Dirichlet noise, and derivation based on branching factor of Checkers.

In training mode select best child node randomly with a probability distribution based on the number of times each child node was visited divided by the total number of visits to the children.  This means that moves that receive the most number of visits during the MCTS search will usually still be played, but other moves have a chance of being selected as well.  Increases exploration during generation of training data.

Temperature Tau variable that gradually tapers off this additional exploration after a user defined number of moves.

## State Representation
Followed DeepMind's lead based on its state for Go and Chess.  Do not need the previous 7 or 8 moves encoded in the state like in Go (Go has special case where previous moves change rules?).  Encode men and kings in different layers for both P1 and P2, have a layer indicating current player, a draw counter layer, and use a similar system as Chess for encoding move.  Luckily, there are only 8 possible moves in Checkers for any given piece (4 moves, 4 jumps in 4 directions) rather than 4000+ in Chess.  One layer for each of the 8 possible moves.

## Initial Neural Network Architecture
DeepMind uses residual NN architecture with skip connections and batch normalization.
### Policy Head
Only 256 possible probs (8 possible moves * 32 possible squares) - half checker board not used.
### Value Head
## Final Neural Net Architecture
[[5]](#references)


## The DeepMind Training Pipeline

### Parallel Processing/Computational limitations

what is goal?  Just show improvement in agent after each successive training iteration
verify against external opponent (flash game)

Experience replay, look this up:
If the network learned only from consecutive samples of experience as they occurred sequentially in the environment, the samples would be highly correlated and would therefore lead to inefficient learning. Taking random samples from replay memory breaks this correlation.

## Discussion of Results
Show results of last iteration of NN vs all previous iterations?
What is the stopping point?  Play versus myself?  Versus another Checkers algorithm?  Is Chinook too good?
https://cardgames.io/checkers/
https://checkers-free.com/
https://www.247checkers.com/
Has difficulty modes
### Future Work

## Acknowledgments

Thanks also to @int8 for his MCTS [blog post](https://int8.io/monte-carlo-tree-search-beginners-guide/) and [source code](https://github.com/int8/monte-carlo-tree-search) which I used as a reference when writing my own MCTS class. 

The wood texture of the Checkers board used in my PyGame GUI is from "5 wood textures" by Luke.RUSTLTD, licensed CC0: https://opengameart.org/content/5-wood-textures.

### References

 1. D. Silver et al., "Mastering the game of Go with deep neural networks and tree search", _Nature_, vol. 529, no. 7587, pp. 484-489, 2016. Available: 10.1038/nature16961 [Accessed 14 December 2020].
 2. D. Silver et al., "Mastering the game of Go without human knowledge", _Nature_, vol. 550, no. 7676, pp. 354-359, 2017. Available: 10.1038/nature24270 [Accessed 14 December 2020].
 3. D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play", _Science_, vol. 362, no. 6419, pp. 1140-1144, 2018. Available: 10.1126/science.aar6404 [Accessed 14 December 2020].
 4. D. Foster, "AlphaGo Zero Explained In One Diagram", _Medium_, 2020. [Online]. Available: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0. [Accessed: 14- Dec- 2020].
 5. C. Browne et al., "A Survey of Monte Carlo Tree Search Methods", _IEEE TRANSACTIONS ON COMPUTATIONAL INTELLIGENCE AND AI IN GAMES_, vol. 4, no. 1, pp. 1-49, 2012. [Accessed 14 December 2020].
 6. "WCDF Official Rules", _The American Checker Federation_, 2020. [Online]. Available: http://www.usacheckers.com/downloads/WCDF_Revised_Rules.doc. [Accessed: 14- Dec- 2020].


## Footnotes
\* PyGame 2.0.0 appears to have some bug that causes Ubuntu to believe that the PyGame GUI is not responding (although it is clearly running).  I recommend using PyGame 1.9.6 if you experience similar issues with newer versions of PyGame.



====================
==#CheckersMCTS==


===Progress Made===
- Finished tweaking PyGame GUI (make sure to use v1.9.6 on laptop Checkers-MCTS environment)
- Started 1000 game training session with 100 move limit
- Started writing readme.md


===Short-term Goals===
- Make 100 move limit an argument (currently hard-coded)
- Finish MCTS section of readme
- Make sure to acknowledge int8 https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/nodes.py
- Have GUI (if NN = True) show player's confidence of winning (Q-value of current state)
- Show percentage chance of choosing each move?  Color code ghosts (Blue <20% chance, yellow 20-50%, green 50+%?)

===Long-term Goals===
- Train NN and start next training iteration (tweak learning rate? Make a list of LRs?)
- Finish writing readme (at least up to final results portion)
- Use unit test for checkers class?  Could have caught the errors in the state sooner if you had


===References===
- https://hci.iwr.uni-heidelberg.de/system/files/private/downloads/306713958/report_philipp_wimmer.pdf
-

===Notes===
+

+

+ 

====================
