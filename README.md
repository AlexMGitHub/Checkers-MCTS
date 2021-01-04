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
[I previously implemented](https://github.com/AlexMGitHub/Minesweeper-DDQN) an agent that learned to play Minesweeper based on DeepMind's work with Atari games.  DeepMind published several papers between 2013 and 2016 that used deep double Q-learning to train neural networks to play a variety of Atari games at better-than-human performance.  That's not to say *super*-human performance, but the agent often achieved a higher score than the human player used as a baseline for the game.  I found that my Minesweeper agent won 23.6% of its games on Expert difficulty.  This is a far higher win rate than what I personally could achieve as a human player, but a "perfect" player would be expected to win more than 35% of its games.  Although DeepMind's trained agents successfully learned to play many of the Atari games, in some of the games the agents performed significantly worse than the human baseline.  This earlier mixed success on Atari games was blown out of the water by the 4-1 victory of [AlphaGo against Lee Sedol](https://deepmind.com/research/case-studies/alphago-the-story-so-far) in March 2016.  


### AlphaGo, AlphaGo Zero, and AlphaZero

*AlphaGo's* [[1]](#references) defeat of an 18-time world Go champion was a major milestone in artificial intelligence.  Go is far more complex than any Atari game with more than 10 to the power of 170 possible board configurations.  Prior to *AlphaGo* and its successors, no Go program had defeated a professional Go player.  *AlphaGo Zero* [[2]](#references) was a refinement of *AlphaGo* that required no human knowledge (no training on games between human players) and became an even stronger player than *AlphaGo*.  Finally, DeepMind generalized the *AlphaGo Zero* algorithm (named *AlphaZero* [[3]](#references)) and applied it to the games of Go, Chess, and Shogi with outstanding results.  DeepMind claims that *AlphaGo* (or its successors) is the greatest Go player of all time.  

Go is a two-player game, whereas the Atari games DeepMind previously worked with were single-player games.  In a single-player game the agent interacts with a game environment and receives rewards or punishments based on those interactions.  In two-player games like Go, the agent instead plays against an opponent and receives a reward or punishment depending on the outcome of the match.  This led to DeepMind training the agent through *self-play*, that is having the agent play millions of games of Go against *itself*.  The experience gathered through these self-play games is used to train a very deep neural network.  Once trained, the neural network plays a series of games against the previous iteration of the network to see which neural network is stronger.  The better neural network then plays many thousands of games against itself to generate a new set of training data, and the entire cycle repeats itself until the agent has reached the desired level of play strength.

*AlphaGo's* neural network does not simply make inferences to decide what moves to play [[4]](#references).  Instead, the neural network's inferences are paired with an algorithm called Monte Carlo Tree Search. 


## Monte Carlo Tree Search

### Overview
I (and most blogs I found) referenced [[5]](#references) for a fantastic overview of the Monte Carlo Tree Search algorithm.  The algorithm creates a tree composed of nodes, each of which represents a possible move or game state.  Every node also stores statistics about the desirability of the potential move that it represents.  The root node represents the current game state, and its child nodes are the legal possible moves that can be made from that root node.  Each child node has children of its own, which are possible moves that the opponent can make.  These "grandchild" nodes have children of their own, which are moves that the original player can make after his opponent has moved, and so on.  The tree quickly grows large even for relatively simple games, and each branch eventually ends in a terminal node that represents one possible game outcome (win, loss, or draw).  The [branching factor](https://en.wikipedia.org/wiki/Game_complexity#Complexities_of_some_well-known_games) of games like Chess and Go are so large that it's not possible to represent every potential move in the tree, and so the tree search is constrained by a user-defined computational budget.

<p align="center">
<img src="docs/images/mcts_phases.png" title="MCTS Diagram" alt="MCTS Diagram" width="640"/>
</p>

Given these computational restraints, the tree search must proceed in such a way as to balance the classic trade-off between exploration and exploitation.  The above image is taken from [[5]](#references) and gives a visual overview of the four phases of the Monte Carlo Tree Search algorithm.  In each iteration of the MCTS algorithm, a *tree policy* decides which nodes to select as it descends through the tree.  Eventually, the tree policy will select a node that has children that have not yet been added to the tree (expandable node).  One of the unvisited child nodes is added to the tree, and a simulation or "rollout" is played out according to a *default policy*.  The simulation starts at the newly added node and continues until it reaches a terminal state.  Finally, the outcome of the simulation is backpropagated up the tree to the root node, and the statistics of each node along the way are updated.  This is a simple algorithm, but what makes it really effective are the two policies: the tree policy and default policy. 


### Tree Policy
The tree policy balances the exploration/exploitation trade-off.  It does this by treating the choice of child node as a multi-armed bandit problem, and uses a variant of the Upper Confidence Bound algorithm called UCT (Upper Confidence bounds applied to Trees).  As mentioned in the previous section, each node stores statistics that are updated whenever a simulation outcome is backpropagated through it.  The UCT formula is:

<p align="center">
<img src="docs/images/UCT_formula.png" title="UCT Formula" alt="UCT Formula" width="300"/>
</p>

Where:

* **q<sub>j</sub>** The total reward or Q-value of child node *j*
* **n<sub>j</sub>** The number of visits to child node *j*
* **X<sub>j</sub>** The average Q-value of child node *j* (q<sub>j</sub> / n<sub>j</sub>)
* **n** The number of visits to the parent node
* **C<sub>p</sub>** A constant of value > 0 used to tune the amount of exploration

Each node in the tree keeps track of its *q* and *n*, and the UCT score of the node may be recalculated for each iteration of the tree policy.  The node with the highest UCT score amongst its "sibling" nodes will be selected for traversal.  Nodes with higher average Q-values (X<sub>j</sub>) will have commensurately higher UCT scores to encourage exploitation.  However, this is offset by the second term that encourages exploration.  The natural log of the total number of visits to the parent node is divided by the number of visits to child node *j*.  If child node *j*'s sibling nodes have received many visits while node *j* has received relatively few, this second term will increase compared to its siblings.  The UCT score of node *j* will also increase and may eventually cause it to be selected by the tree policy.  This means that nodes with low average Q-values still get selected occasionally to ensure that they really are sub-optimal choices and not just unlucky.  Without this exploration term, a move that could be optimal but had some initial rollouts that lowered its average Q-value would never be visited again.  DeepMind used a modified version of UCT for *AlphaZero* that will be discussed in a later section.


### Default Policy

As the tree policy descends down the tree it will eventually encounter an expandable node.  An expandable node is a node that has unvisited child nodes that have not yet been added to the tree.  The tree policy adds one of the unvisited child nodes to the tree, and then performs a simulation from that new node according to the default policy.  The default policy for standard MCTS is to simulate the remainder of the game by simply choosing random moves for both players until the game reaches a terminal state.  The reward based on the simulated outcome of the game is then backpropagated through the tree starting with the new node and ending with the root node.  Each node in the selected path has its *n* value incremented by one, and its *q* value increased by +1, 0, or -1 for a win, draw, or loss, respectively.  The next iteration of the tree policy uses these updated statistics to calculate the nodes' UCT scores.

Simulating the game's outcome by randomly choosing actions may at first seem like a poor way to estimate the value of a potential move.  In effect, this default policy estimates the probability of winning from a given state by randomly sampling the possible outcomes from that state.  If enough random rollouts are performed then by the law of large numbers the resulting average Q-value of the node should approach the expected value of the probability distribution.  Stronger moves will have a larger expected value (higher chance of winning), and weaker moves will have a smaller expected value (lower chance of winning).  These random rollouts have apparently been used with a degree of success even in fairly complex games like Chess.  For games like Checkers with small branching factors a modest number of rollouts can result in a decent level of play.  For games like Go with enormous branching factors random rollouts will be far less effective.  

*AlphaZero* completely removed random rollouts and replaced them with so-called "truncated rollouts."  When the tree policy reaches an expandable node, the default policy inputs the node's state to the neural network and the resulting estimated Q-value is backpropagated.  More details will be presented in a later section.      

### Choosing the Best Child
The tree search ends once the computational budget is exhausted.  The root node's children will now have statistics associated with them that can be used to determine the best next move.  This "best child" can be chosen according to several different criteria according to [[5]](#references):

1. Node with the highest reward (Max child)
2. Node with the most visits (Robust child)
3. The node with the highest visit count and highest reward (Max-Robust child)
4. The child which maximizes a lower confidence bound (Secure child)

According to [[5]](#references) the robust child method is the most commonly used criterion.  Once a player has selected its move, the resulting state becomes the root node of the opposing player's tree search.  All of the nodes representing the possible moves that the first player *didn't* make are discarded (or pruned) from the second player's tree.  Only the nodes representing the first player's chosen move (and any child nodes) are preserved.  The statistics of these preserved nodes will be used in the second player's new tree search.


### MCTS Implementation
A critical "gotcha" that is not well-explained in [[5]](#references) is how to determine a node's reward based on the outcome of the game.  Some of the nodes in the tree represent player 1's potential moves, and the other node's represent player 2's possible moves.  If player 1 wins the game in the simulation, nodes representing its potential moves should receive a reward of +1.  Player 2 then lost the simulated game and should receive a reward of -1 to all nodes representing its potential moves.  

This seems straightforward, and yet when I ran my Tic-Tac-Toe game environment to test my MCTS implementation both players made nonsensical moves.  Rather than attempting to win (or prevent the other player from winning) both players seemed to choose the *worst* possible move in every situation!  It took some time to understand the mistake that I had made.  Originally, if player 1 won the game I awarded +1 to all nodes with player 1 as the current player, and all nodes with player 2 as the current player a -1.  I eventually realized that this is exactly backwards:

If the player of the current state is player 1, that means that the current node is a child node (potential move) of player 2.  Therefore the rewards are flipped: a player 1 victory on a player 1 node receives a reward of -1.  A player 2 victory on a player 1 node receives a reward of +1.  This adjusts the UCT value of the node from the opponent's perspective, and incentivizes the MCTS to choose nodes that favor player 2 when representing it during the selection phase.  The same is true for player 2 nodes so that they are chosen from player 1's perspective during the selection process.

After making this change, the Tic-Tac-Toe agents played optimally and every game ended in a draw.  However, there was one further wrinkle when implementing MCTS for the game of Checkers.  In the event of a multi-jump (double-jump, triple-jump, etc.) the player is required to make more than one move per turn.  This means that either player could make two or more moves in a row.  To handle this scenario I explicitly check the parent node's player and compare it to the outcome of the simulation.  If player 1 wins the simulation and the current node's parent node is also player 1 then the reward is +1.  Otherwise, the reward is -1.  

One final headache was implementing the MCTS reward for a neural network's "truncated rollout" rather than a random rollout.  A truncated rollout is named such because it does not simulate the outcome of the game.  Instead, the neural network outputs a Q-value estimate for the node's state, and that value is backpropagated through the tree.  Because the state representation of the board does not contain information about previous moves (see relevant section below), the neural network doesn't know who the previous player was when estimating the Q-value of a given state.  The neural network can only estimate the Q-value of the state from the perspective of the current player.  I handled this in the MCTS code by comparing the parent node's player to the root node's player, and if they match the neural network's Q-value estimate is added to the current node's total reward statistic unchanged.  If the root node's player doesn't match the parent node's player then the neural network's estimate is multiplied by -1 prior to adding it to the current node's total Q-value statistic.  This then assigns the reward to each node according to the perspective of the player who chose it.    


### Parallel Processing

DeepMind used a highly sophisticated asynchronous parallel processing scheme to dramatically increase computational throughput.  As I only have access to a single laptop with a single GPU, I chose to proceed without attempting to recreate DeepMind's parallelized training pipeline.  In addition, Python's Global Interpreter Lock (GIL) prevents truly parallel multithreading.  I did experiment with using Python's ``multiprocessing` module to run random rollouts in parallel, but found the speed-up to be negligible.  It's possible that if I had used DeepMind's "virtual loss" approach I might have had better luck, but as I only have a single GPU for neural network inferences it ultimately did not seem worth the trouble to investigate further.  The multiprocessing option is available for use with random rollouts.
   

### Validating MCTS with Tic-Tac-Toe
Shown below is a Tic-Tac-Toe game in progress.  Player 1 (X) began the game by taking the center, which is the optimal move to make.  Player 2 (O) responded by taking the top-right corner, after which player 1 took the top-center square.  Player 2 must now block the bottom-center square or player 1 will win on its next turn.  The console output below indicates that it is player 1's turn, and that the MCTS completed after reaching its computational budget of 1000 rollouts.  The diagram beneath this information represents the first two layers of the MCTS tree.  The first entry is the root node, where the numbers (-525/1053) represent the root node's total Q-value and number of visits, respectively.  The indented quantities below the root node are its child nodes, also with their total Q-values and number of visits listed.  The child nodes represent player 1's possible moves.  Player 1 chose the top-center square per the robust child criterion as it had the most number of visits.  


````
It's now Player 1's turn (X)
Starting search!
Stopped  search after 1000 rollouts and 00:01.13 duration!
|- (-525/1053) (25.1%)
	|- (54/125) (71.6%)
	|- (175/290) (80.2%)
	|- (45/112) (70.1%)
	|- (152/253) (80.0%)
	|- (2/38) (52.6%)
	|- (51/120) (71.2%)
	|- (47/114) (70.6%)
╒═══╤═══╤═══╕
│ . │ X │ O │
├───┼───┼───┤
│ . │ X │ . │
├───┼───┼───┤
│ . │ . │ . │
╘═══╧═══╧═══╛
````

The signs of the total Q-values are reversed between the root node and its children.  This is because the board state before player 1 made its move was a child node of player 2, and the total Q-value of the root node is thus computed from player 2's perspective.  All of player 1's possible moves indicate a better than 50 percent chance of winning as shown in the second set of parentheses.  This results in a positive total Q-value because the rollouts from that move returned more +1s than -1s.  Player 2's chances of winning from this state then must be worse than 50% and thus the root node is a negative number as the rollouts returned more -1s than +1s.  The root node's corresponding percentage of 25.1% indicates player 2's confidence in winning from that root node state.

````
It's now Player 2's turn (O)
Starting search!
Stopped  search after 1000 rollouts and 00:01.27 duration!
|- (341/1033) (66.5%)
	|- (-39/54) (13.9%)
	|- (-39/54) (13.9%)
	|- (-41/60) (15.8%)
	|- (-36/46) (10.9%)
	|- (-141/744) (40.5%)
	|- (-46/74) (18.9%)
╒═══╤═══╤═══╕
│ . │ X │ O │
├───┼───┼───┤
│ . │ X │ . │
├───┼───┼───┤
│ . │ O │ . │
╘═══╧═══╧═══╛
````

The console output above shows that player 2 blocked player 1's attempt at three in a row, as expected.  Notice in the tree diagram that Player 2 appeared to select the option with the worst total Q-value (-141).  However, the *average* Q-value (total Q / number visits) is the quantity used in the UCT formula.  The bottom-center square has a total Q-value of -141, but also has 744 visits.  Therefore its average Q-value is much larger (smaller magnitude negative number) than the other options.  This is results in the bottom-center node having the highest chance of winning (40.5%) of all other options.  

Note also that the root nodes of these diagrams have more than 1000 rollouts, even though the computational budget is set to 1000 rollouts.  These additional root node visits are from MCTS searches made during previous turns.  As mentioned in a previous section, node statistics are preserved between tree searches for nodes that represent the moves that the players actually chose to make.  

````
It's now Player 1's turn (X)
Starting search!
Stopped  search after 1000 rollouts and 00:00.20 duration!
|- (0/1450) (50.0%)
	|- (0/1449) (50.0%)
╒═══╤═══╤═══╕
│ O │ X │ O │
├───┼───┼───┤
│ X │ X │ O │
├───┼───┼───┤
│ X │ O │ X │
╘═══╧═══╧═══╛
Game over! The outcome is: draw
````

Tic-Tac-Toe is a simple game where optimal play from both players will always result in a draw.  The game does indeed result in a draw, and the total Q-value of player 1's final move is zero - the reward for a draw.  This corresponds to a likelihood of winning of exactly 50% for both the root node (player 2's perspective) and the child node (player 1's perspective).  These results appear to validate that the MCTS algorithm is working as expected.


## DeepMind's Implementation of AlphaZero
The graphic [[4]](#references) created by David Foster provides a great overview of the *AlphaGo Zero* algorithm and its neural network architecture.  *AlphaGo Zero's* neural network architecture is nearly identical to that of *AlphaZero's*, and is comprised of a deep neural network with 40 residual layers followed by two "heads" or outputs.  The residual layers consist of convolutional layers, batch normalization, ReLu activations, and a "skip connection".  The skip connection is key to avoiding the vanishing gradient problem in very deep networks.

When a state is input into the neural network, the value head outputs the estimated value of the current state as a value between -1 and +1.  The policy head outputs a vector of probabilities between 0 and 1 for each possible legal move.  The value head indicates how confident the player is of winning the game from the current state, with a value of +1 indicating 100% confidence in winning and -1 indicating 0% confidence in winning.  The policy head output represents the prior probabilities that a good player would make each of the legal possible moves.  Higher probabilities thus indicate potentially stronger moves.

DeepMind's MCTS algorithm uses both of these outputs when performing rollouts.  As mentioned earlier, *AlphaZero* uses the value head output as a truncated rollout.  Rather than rolling out to a terminal state, the value head output serves as an estimate of the current node's Q-value.  This value estimate is backpropagated and used in a modified UCT formula by the tree policy.  The prior probabilities produced by the policy head are also used in this modified UCT formula.

### AlphaZero's Tree Policy
As with standard MCTS, *AlphaZero's* tree policy selects the child with the maximum UCT score.  However, DeepMind modified the formula to account for the prior probabilities produced by the policy head of the neural network.  This modified formula is referred to as PUCT: Predictor + UCT.  The action *a* selected at a time-step *t* is:

<p align="center">
<img src="docs/images/at_formula.png" title="Action Formula" alt="Action Formula" width="320"/>
</p>

Where Q(*s<sub>t</sub>, a*) is the mean action value of a particular node and and U(*s<sub>t</sub>, a*) is defined as:

<p align="center">
<img src="docs/images/Usa_formula.png" title="U(s, a) Formula" alt="U(s, a) Formula" width="320"/>
</p>

This doesn't look too different from the standard UCT formula other than the P(*s, a*) term.  The average Q exploitation term is still added to the exploration term U(*s<sub>t</sub>, a*).  The exploration term still has an exploration constant *c*, and the number of visits to the parent node is divided by the number of visits to the child node of interest.  There's no natural log function and the scalar terms have been subsumed into *c*, but otherwise the formula provides the same functionality as UCT.  The critical difference is that the P(*s, a*) term adds stochasticity to the PUCT formula:

<p align="center">
<img src="docs/images/Psa_formula.png" title="U(s, a) Formula" alt="U(s, a) Formula" width="320"/>
</p>

Where:

* **&rho;<sub>a</sub>** is the estimated prior probability of action *a*
* **&eta;** ~ Dir(&alpha;)
* **&epsilon;** = 0.25

Dir(&alpha;) refers to the [Dirichlet probability distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution).  DeepMind stated that its goal in adding Dirichlet noise was to encourage more exploration when selecting nodes.  The neural network's assessment of the prior probabilities of the potential moves can (and will) be wrong.  By adding adding the Dirichlet noise to the prior probabilities, a potentially optimal move that the neural network has assigned a lower probability to may get chosen more often.  Another reason for adding noise to the prior probabilities is that neural networks are deterministic.  If no stochastic mechanism is added to the tree policy, then the trained neural networks will play the same game against each other over and over again in the network evaluation phase.  I discovered this first-hand when testing an early version of my tournament code pitting two trained networks against each other for 10 games.  The two trained networks played the exact same game with the same outcome ten times in a row! 

#### The Effect of Alpha
The alpha argument in the Dirichlet function is a vector of scalars of the same value (e.g. \[0.3, 0.3, 0.3, 0.3\]).  The length of the vector is equal to the number of legal moves, and the value of the scalar is inversely proportional to the approximate number of legal moves in a typical game position.  Most people have interpreted the latter statement to be equivalent to the average branching factor of the game.  DeepMind chose alpha to equal 0.3, 0.15, and 0.03 for Chess, Shogi, and Go, respectively.  These games have [average branching factors](https://en.wikipedia.org/wiki/Game_complexity) of 35, 92, and 250, respectively.  The output of the Dir(&alpha;) function is a vector of random numbers sampled from the Dirichlet distribution.  The output vector will be the same length as the input vector, and the sum of the random numbers will equal 1 (e.g. \[0.43, 0.27, 0.22, 0.08\]).  

Aditya Prasad [notes that](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5) the alpha values DeepMind chose are roughly equal to 10 divided by the branching factor of the game.  Using this rule of thumb for the game of Connect Four results in a value of 2.5, but Prasad ultimately found that 1.0 was the optimal value to use during training.  Checkers is listed as having a branching factor of 2.8, which is lower than the branching factor of Tic-Tac-Toe!  Using Prasad's rule of thumb for Checkers results in an alpha of 3.6, but I ended up also using a value of 1.0 after considering the effect of alpha on the noise values. 

DeepMind likely chose the Dirichlet distribution because its output vector of random numbers will always sum to 1.  This guarantees that P(*s, a*), the weighted sum of the prior probabilities and the noise, will also always sum to 1.  This is a useful property when dealing with probabilities.  Although the sum of the noise will always equal 1, the value of alpha controls how uniformly the noise is distributed amongst the potential moves.  Prasad's blog post has a nice visualization of how the alpha value affects both the Dirichlet noise and the resulting P(*s, a*) value.  Large values of alpha "flatten" the noise, and make the values drawn from the Dirichlet distribution more similar in magnitude.  Small values of alpha cause one of the noise values to be "emphasized" over the others, meaning its magnitude is much larger than the other noise magnitudes drawn from the distribution.

Based on this understanding of the Dirichlet distribution, using a large alpha value isn't useful for our purposes.  Adding roughly the same magnitude of noise to all of the prior probabilities just results in a P(*s, a*) that is almost identical to the original prior probabilities.  If more exploration is our goal, then it makes sense to add a "lopsided" amount of noise to the prior probabilities in the hopes that it results in less-favored moves being selected more often.  I found that the average number of moves made during my training games was closer to 6 than to the listed branching factor of 2.8.  I suspect that an alpha value of 1.0 is probably closer to optimal than the rule-of-thumb value of 3.6.  I also used Prasad's suggested value of 4 for the exploration constant *c*.

### Self-Play
#### AlphaZero's Best Child Selection

  


### Training the Network

### Evaluating the Network




## Checkers Mechanics

  or even chess.game than The Atari games are single player, Go/Shogi/Chess are two player (self-play).
Chose Checkers as a simpler game that may be possible to get good performance with far less computation, but still complicated enough not to be trivial (like Tic-Tac-Toe).
branching factor requires nns to narrow down search

When jumps are possible they are mandatory.  Multiple jumps are allowed in one turn.


## The DeepMind Training Pipeline

### Adding Stochasticity
Neural networks are deterministic, will play the same moves in the same situation.

Add Dirichlet noise to neural network prior probabilities prediction.  (Show UCT formula).  Means that MCTS will tend to play out differently for the same state, which means that games between the same neural network are no longer deterministic and will tend to differ.  Show equation with sigma and U(s,a).  Discuss value for C - didn't bother using variable C, but why choose particular constant?  Also discuss alpha value for Dirichlet noise, and derivation based on branching factor of Checkers.

In training mode select best child node randomly with a probability distribution based on the number of times each child node was visited divided by the total number of visits to the children.  This means that moves that receive the most number of visits during the MCTS search will usually still be played, but other moves have a chance of being selected as well.  Increases exploration during generation of training data.

Temperature Tau variable that gradually tapers off this additional exploration after a user defined number of moves.

Dirichlet noise 
https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5

To pick a value, we first looked at the values they used for chess, shogi, and Go: 0.3, 0.15, and 0.03. The average number of legal moves in those games are 35, 92, and 250. It’s hard to know how to optimally extrapolate, but a reasonable first guess looks to be choosing ɑ = 10/n. Since there are approximately four legal moves in a Connect Four position, this gives us ɑ=2.5. It’s true that this is greater than one while the rest are less, and yet this number seemed to do well in our testing. With a little playing around, we found that 1.75 did even better.

Branching factor of checkers listed as 2.8, which would lead to a Alpha of 3.6.  Taking average number of moves from training data got a number closer to 6, which would result in alpha of 1.7.  In blog post above found 1.75 and then later 1.0 did even better than the expected 2.5.  Lower alpha means noise applied less equally to options.  Could be that the roughly equal amount of noise applied when using large alpha values isn't useful because it rarely overcomes the move with the highest prior prob.

## State Representation
Followed DeepMind's lead based on its state for Go and Chess.  Do not need the previous 7 or 8 moves encoded in the state like in Go (Go has special case where previous moves change rules?).  Encode men and kings in different layers for both P1 and P2, have a layer indicating current player, a draw counter layer, and use a similar system as Chess for encoding move.  Luckily, there are only 8 possible moves in Checkers for any given piece (4 moves, 4 jumps in 4 directions) rather than 4000+ in Chess.  One layer for each of the 8 possible moves.

## Initial Neural Network Architecture
DeepMind uses residual NN architecture with skip connections and batch normalization.
discuss loss times (MSE, kullback), loss weights, batch normalization, etc.
https://stackoverflow.com/questions/48620057/keras-loss-weights
### Policy Head
Only 256 possible probs (8 possible moves * 32 possible squares) - half checker board not used.
### Value Head
## Final Neural Net Architecture
[[5]](#references)


## The DeepMind Training Pipeline

Train using average of z and q:
https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628

DeepMind trained the value head on *z*, the reward based on the outcome of the game from the perspective of the current player.  The value of *z* can only be -1, 0, or +1 (loss, draw, win) and this label is assigned to all of the player's states in that particular game.  This means that if the player won his game, even poor moves that the player made during the game would be assigned a value of +1.  Conversely, if the player lost the game even the good moves the player made are assigned a value of -1.  With enough training data this "mislabeling" averages out, but for the smaller amounts of training data available to me it could be an issue.  An alternative training target could be the MCTS q-value of the state.  The advantage of using *q* is the valuation of the state after many MCTS rollouts.  The end result of the game doesn't matter; just the result of the simulations.  As discussed in Vish Abram's blog post, using *q* as the training target has two issues: the first, and most obvious issue is that the initial generation of neural networks has no idea how to evaluate a position, and so the q-value of the MCTS node is worthless.  Second, with a limited number of rollouts the q-value may not take into account critical moves made just beyond the rollout "horizon."

Abrams shows the results of training on *z*, training on *q*, training on the average of *z* and *q*, and training on *z* with a linear falloff towards *q* over several generations of training data.  He states that the latter gave him the best results, however his plot seems to indicate that averaging *z* and *q* actually provided better results; in any case the performance was very close.

   
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

Thanks to @int8 for his MCTS [blog post](https://int8.io/monte-carlo-tree-search-beginners-guide/) and [source code](https://github.com/int8/monte-carlo-tree-search) which I used as a reference when writing my own MCTS class. 

Thanks to Rajat Garg for his [blog post](https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71) explaining how to create a Keras generator to handle datasets that are too large to fit in GPU memory.

Thanks to https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5

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