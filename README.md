# Nemesis-Engine
#### Modeling rapid response to child abductions using multi-agent adversarial networks with Reinforcement Learning. 

![animation-ezgif com-resize](https://github.com/user-attachments/assets/d29810d4-77b8-4986-9c04-c72f8be14ed0) 




----------------------------

## Problem Statement 

"Child abductions by strangers" are one of the most dangerous crimes that law enforcement agencies like the FBI investigate. This is because this class of crime comes with it a rapidly shrinking response window:

**First responders typically have around 90 minutes to ascertain where an abductor and victim might be.**

**They then have another 90 minutes to carry out a rescue attempt.**

**After this critical 3-hour mark, around 75% of victims are never found again.**

-----------------------------


## The Nemesis Engine

The Nemesis Engine is primarily a strategic tool. Its a module that allows for an investigation into how law enforcement can be mobilized more effectively across real cities and towns. 

The engine simulates, or models, a game of "hide and seek". It pits a team of N "seekers" against one "hider" with relatively simple rules. If the hider can evade the seekers for the length of the simulation (90-180 minutes), the hider wins. Otherwise, if any one of the seekers finds the hider, then the seeker team wins. 

Through reinforcement learning, both the seekers and the hider become more skilled at thei respective roles. The team of seekers become better first-responders; the hider becomes a better evader.

If the seekers can manage to consistently find the hider -  even when the hider becomes highly skilled at evading detection, - then we may have identified ways for teams of first responders to mobilize in respond to abduction events. 


-------------------------------------------------

## The Build Process

The generic build process for this simulation is noted as follows: 

1. Build a "digital twin" of a geographic region
2. Develop a way to place agents dynamically in the environment.
3. Construct a global value function (Critic or Q-function)
4. Construct a local policy function (Actor)
5. Simulate agent actions through Monte Carlo Tree Search (MCTS)
   - Save all intermediate environment states - local MCTS policies and global game states
   - Append simulation outcomes to global game states
6. Train Global Value Function against simulation game states.
7. Train Local Policy Function against MCTS derived policies

Steps 5-7 can then be cycled over and over until diminishing returns are had in agent self-improvement.

------------------------------------------------------


# Theoretical Framework

The framework of this engine is largely inspired by Google Deepmind's seminal work with AlphaZero and AlphaGo. These were algorithms that taught themselves how to play board games like chess and Go without any human guidance or intervention.

You can read one of the original papers here: [Original Publication](https://sci-hub.ru/10.1038/nature24270) .

We use a modified version of the Monte Carlo search process that Deepmind outlined for their methodology in self-play reinforcement learning. 

This process can be divided up into two distinct parts:

   1. Self-play: the computer takes up both sides of the game simultaneously and plays against itself using its best judgement of its past experience playing the game. It does not update its knowledge during this phase.
   2. Training: every game state during the self-play is saved in and collected. Two deep neural networks are trained off these game states to update the algorithm's "knowledge" of the game.

The two deep neural networks that make up the "knowledge base" of the algorithm are as follows:

 - A global value estimator: a function that takes in an environment state vector and outputs a value between (-1,1). You can lazily interpret this value as the probability of failure/success of the current game state. 
 - A local policy header: a function that takes in agent's observation and outputs a probability distribution across the valid choices an agent can make at that given timestep.

Let's examine each piece in isolation.

## Self-Play through modified Monte Carlo Tree Search

In two-player board games, the sequence of actions taken by each side can be modeled as a game tree, where each layer corresponds to one player’s decisions, and the layer below represents the opponent’s responses. Consider the simple case of Tic-Tac-Toe:

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/7c4d391d-bee1-4976-b1b7-7d5a3f5ecfd3" />


When the state space is small, “solving” the game is straightforward: you can explore the entire tree to its terminal nodes and identify the optimal move in every situation. In Tic-Tac-Toe, an algorithm like MinMax can brute-force the entire tree, guaranteeing perfect play.


<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/0ec5207a-3a77-4f31-9189-2df055ebc6db" />

For more complex games like chess, this approach collapses under the weight of combinatorial explosion. The number of possible move sequences grows exponentially with each ply, making it impossible to explore the full tree. Even with heavy pruning and heuristics, MinMax-style engines are typically limited to ~15–20 moves ahead, with 30–40 moves being an upper bound in specialized setups.

Monte Carlo Tree Search (MCTS) addresses this challenge by trading exhaustive exploration for probabilistic sampling. Instead of attempting to evaluate every branch, MCTS uses a tree policy to decide which branches to explore more deeply, balancing moves that look promising with those that haven’t been tried enough yet.

The most common tree policy is the **Upper Confidence Bound for Trees** (UCB1):

```math
a^* = \arg\max_a \left[ Q(s, a) + C \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]
```

Where: 

| Symbol       | Definition                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `a`          | Candidate action from the current state `s`.                               |
| `a^*`        | Action selected by the search policy (argmax output).                      |
| `Q(s, a)`    | **Empirical mean return** (average reward) for taking action `a` in state `s`. |
| `N(s)`       | Number of times state `s` has been visited.                                 |
| `N(s, a)`    | Number of times action `a` has been taken from state `s`.                   |
| `W(s, a)`    | Cumulative sum of rewards obtained after taking `a` from `s` (used to compute `Q`). |
| `C`          | **Exploration constant** for UCT; balances exploration vs exploitation.     |
| `\ln N(s)`   | Natural logarithm of state visit count (used in the exploration term).      |


This formula has two parts:

- **Exploitation (Q)** : Steers the search toward actions with higher observed rewards.
- **Exploration** : The square-root term biases the search toward under-visited moves to gather more information.
  
Early in the search, the exploration term dominates, ensuring a wide sampling of possibilities. As the tree fills out, the exploitation term becomes more precise, and exploration pressure naturally fades — allowing the algorithm to focus on the most promising branches.


