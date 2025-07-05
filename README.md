# Nemesis-Engine
#### Modeling rapid response to child abductions using multi-agent adversarial networks with Reinforcement Learning. 

![animation-ezgif com-resize](https://github.com/user-attachments/assets/d29810d4-77b8-4986-9c04-c72f8be14ed0) 




----------------------------

## Problem Statement 

"Child abductions by strangers" are one of the most dangerous crimes that law enforcement agencies like the FBI investigate. This is because this class of crime comes with it a rapidly shrinking response window:

**First responders typically have around 90 minutes to ascertain where an abductor and victim might be.**

**They then have another 90 minutes to carry out a rescue attempt.**

**After this critical 3-hour mark, around 75% of victims are either never found again.**

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


## Theoretical Framework

The framework of this engine is largely inspired by Google Deepmind's seminal work with AlphaZero and AlphaGo. These were algorithms that taught themselves how to play board games like chess and Go without any human guidance or intervention.

You can read one of the original papers here: [Original Publication](https://sci-hub.ru/10.1038/nature24270) .

We use a modified version of the Monte Carlo search process that Deepmind outlined for their methodology in self-play reinforcement learning. 

This process can be divided up into two distinct parts:

   1. Self-play: the computer takes up both sides of the game simultaneously and plays against itself using its best judgement of its past experience playing the game. It does not update its knowledge during this phase.
   2. Training: every game state during the self-play is saved in and collected. Two deep neural networks are trained off these game states to update the algorithm's "knowledge" of the game.

The two deep neural networks that make up the "knowledge base" of the algorithm are as follows:

 - A global value estimator: a function that takes in an environment state vector and outputs a value between (-1,1). You can lazily interpret this value as the probability of failure/success of the current game state. 
 - A local policy header: a function that takes in agent's observation and outputs a probability distribution across the valid choices an agent can make at that given timestep.



