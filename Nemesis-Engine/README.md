# Nemesis-Engine
Modeling child abduction response by law enforcement using multi-agent adversarial networks. 

----------------------------

## Introduction 

The FBI deems "child abductions by strangers" one of the most dangerous crimes their agents face. This is because in the vast majority of cases, a responder only has around 90 minutes to track down the abductor. 

After this time horizon, the recovery rate drops precipitously - in 75% of cases the child is never seen again. 

Analytical tools that can parse an Amber Alert and immediately convert the disparate signals into meaningful insights might be a useful tactical device - (see: Project Amber) - but understanding where a suspect might be is only half the battle. 

Suppose we had a crystal ball which could reveal to us the general location of a suspected child abductor and the corresponding victim. We'll label this general area the "hot zone". 

How does law enforcement mobilize in response to this insight? 

- Are patrol cars dispatched to a single point? 

- Are they dispatched to multiple points? 

- Does one team go door to door in neighborhoods while another team zips from motel to hotel to bed-and-breakfast? 


Even with a coordinated team of five or more patrol units, it isn't so easy to canvas an entire region quickly in order to find a hidden target. To make matters worse, it isn't guaranteed that the target is static or stationary. The abductor could be actively evading law enforcement. The abdutor could be fleeing, not only the "hot zone", but the county and state themselves. 

-----------------------------


## The Nemesis Engine

The Nemesis Engine is a strategic tool that investigates how a rapid response team can be mobilized in order to maximize the chances of stopping an abduction in progress. 

Its a simulation-based method that models a game of "hide and seek" across real road networks and city features. It pits a team of N "seekers" against one "hider" with relatively simple rules. If the hider can evade the seekers for the length of the simulation (90 minutes), the hider wins. Otherwise, if any one of the seekers finds the hider, then the seeker team wins. 

Through reinforcement learning, both sides of the game get more skilled over many repeated generations and simulations. The team of seekers become better first-responders; the hider becomes a better evader.


If the seekers can manage to consistently find the hider -  even when the hider becomes highly skilled at evading detection, - then we may have identified ways for teams of first responders to mobilize in respond to abduction events. 


-------------------------------------------------

## The Build Process

The generic process for building the Nemesis Engine was inspired by Deepmind's seminal algorithm - AlphaZero. It uses a modified Monte Carlo Tree Search to simulate agent actions and derive training data for both the local agent policies and the global value function. 

**Each of these steps can be explored in more detail through the .ipynb notebook files in /BuildProcess/.**

1. Construct a "digital twin" of a geographic region
2. Establish spawn points for your team of seekers and your hider
3. Construct a global value function (Critic or Q-function)
4. Construct a local policy function (Actor)
5. Simulate agent actions through Monte Carlo Tree Search (MCTS)
   - Save all intermediate environment states - local MCTS policies and global game states
   - Append simulation outcomes to global game states
6. Train Global Value Function against simulation game states.
7. Train Local Policy Function against MCTS derived policies
