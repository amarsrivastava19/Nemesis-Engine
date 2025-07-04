# Nemesis-Engine
Modeling child abduction response by law enforcement using multi-agent adversarial networks with reinforcement learning. 

<center> ![animation-ezgif com-resize](https://github.com/user-attachments/assets/d29810d4-77b8-4986-9c04-c72f8be14ed0) </center>



----------------------------

## Problem Statement 

"Child abductions by strangers" are one of the most dangerous crimes that law enforcement agencies like the FBI investigate. This is because this class of crime comes with it a rapidly shrinking response window:

**First responders typically have around 90 minutes to ascertain where an abductor and victim might be.
They then have another 90 minutes to carry out a rescue attempt. 
After this critical 3-hour mark, around 75% of victims are either never found again.**

-----------------------------


## The Nemesis Engine

The Nemesis Engine is a strategic tool that investigates how a rapid response team can be mobilized in order to maximize the chances of stopping an abduction in progress. 

Its a simulation-based method that models a game of "hide and seek" across real road networks and city features. It pits a team of N "seekers" against one "hider" with relatively simple rules. If the hider can evade the seekers for the length of the simulation (90 minutes), the hider wins. Otherwise, if any one of the seekers finds the hider, then the seeker team wins. 

Through reinforcement learning, both sides of the game get more skilled over many repeated generations and simulations. The team of seekers become better first-responders; the hider becomes a better evader.

If the seekers can manage to consistently find the hider -  even when the hider becomes highly skilled at evading detection, - then we may have identified ways for teams of first responders to mobilize in respond to abduction events. 


-------------------------------------------------

## The Build Process

The generic process for building the Nemesis Engine was inspired by Deepmind's seminal algorithm - AlphaZero. It uses a modified Monte Carlo Tree Search to simulate agent actions and derive training data for both the local agent policies and the global value function. 

1. Construct a "digital twin" of a geographic region
2. Establish spawn points for your team of seekers and your hider
3. Construct a global value function (Critic or Q-function)
4. Construct a local policy function (Actor)
5. Simulate agent actions through Monte Carlo Tree Search (MCTS)
   - Save all intermediate environment states - local MCTS policies and global game states
   - Append simulation outcomes to global game states
6. Train Global Value Function against simulation game states.
7. Train Local Policy Function against MCTS derived policies

------------------------------------------------------


## Theoretical Framework
