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

.

.


## Self-Play through modified Monte Carlo Tree Search

In two-player board games, the sequence of actions taken by each side can be modeled as a game tree, where each layer corresponds to one player’s decisions, and the layer below represents the opponent’s responses. Consider the simple case of Tic-Tac-Toe:

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/7c4d391d-bee1-4976-b1b7-7d5a3f5ecfd3" />


When the state space is small, “solving” the game is straightforward: you can explore the entire tree to its terminal nodes and identify the optimal move in every situation. In Tic-Tac-Toe, an algorithm like MinMax can brute-force the entire tree, guaranteeing perfect play.


<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/0ec5207a-3a77-4f31-9189-2df055ebc6db" />

For more complex games like chess, this approach collapses under the weight of combinatorial explosion. The number of possible move sequences grows exponentially with each ply, making it impossible to explore the full tree. Even with heavy pruning and heuristics, MinMax-style engines are typically limited to ~15–20 moves ahead, with 30–40 moves being an upper bound in specialized setups.

Monte Carlo Tree Search (MCTS) addresses this challenge by trading exhaustive exploration for probabilistic sampling. Instead of attempting to evaluate every branch, MCTS uses a tree policy to decide which branches to explore more deeply, balancing moves that look promising with those that haven’t been tried enough yet.

.


<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/7f6a9fb2-b322-469a-8fd2-9a2e5a6f1f79" />

.

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

Deepmind's seminal work with *AlphaZero* and *AlphaGo Zero* adapted this tree policy to include deep neural network functions - so that just given a board state and individual observations for an agent, the tree policy can be estimated without the need for doing exhaustive rollouts down the tree and backpropogation of values to update the MCTS policies. The two networks represent a sort of knowledge base that an agent can rely on to intuit which moves likely lead to the best rewards. 

.

This policy, termed **Predictor + Upper Confidence Bound** (PUCT) is defined as follows:


```math
a^* = \arg\max_a \left[ Q(s, a) + C_{puct} \cdot P(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]
```


Where: 


| Symbol         | Definition                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `a`            | Candidate action from the current state `s`.                               |
| `a^*`          | Action selected by the search policy (argmax output).                      |
| `Q(s, a)`      | **Empirical mean return** for action `a` in state `s`, updated through backpropagation. |
| `N(s)`         | Number of times state `s` has been visited.                                 |
| `N(s, a)`      | Number of times action `a` has been taken from state `s`.                   |
| `P(s, a)`      | **Policy prior** from the neural network; predicts which moves are promising. |
| `C_{puct}`     | **Exploration constant** for PUCT; controls how much the prior affects exploration. |
| `\sqrt{N(s)}`  | Square root of total state visits, used to scale the exploration term.     |
| `1 + N(s, a)`  | Denominator that tempers the influence of `P(s,a)` as an action becomes heavily explored. |


The algorithm doesn’t brute-force the tree all the way to terminal states. Instead, it **expands one new node at a time** in the next layer of the search tree. For each expansion, the **value network** estimates the expected reward of that node, and the algorithm **backpropagates** the resulting value and visit counts up the tree.  

This process repeats — **one node at a time** — while the search tree’s visit counts are continuously updated and stored in memory. Once the agent commits to a move, the visit counts are **softmax-normalized** and saved. These normalized counts are later used to **train the policy network**, so that given a raw observation, the network can predict what the MCTS search *would* recommend if it had run its simulations.  

During gameplay, every turn’s **game state, observation vectors, and other relevant metadata** are logged, creating a library of intermediate states.  

When the game concludes, the **final outcome** (win/loss/draw) is appended to each saved state. This “self-play” loop generates rich training data for both networks:  
- The **policy network** improves at predicting which actions MCTS would favor.  
- The **value network** gets better at distinguishing promising states from poor ones, helping guide which nodes the search should expand next.  


## Modified MCTS for Abduction Simulations


This **neural network–assisted MCTS** is a natural fit for abduction simulations because the search space is massive. Even a *low-resolution* road network of the Des Moines metro area contains roughly **32,000 possible edges**, each representing an action that a hider or seeker might take. Move to a *high-resolution* network — one that includes residential streets, alleyways, and undeveloped roads — and that number can easily exceed **100,000**.

Fortunately, we’re somewhat insulated from this overwhelming branching factor by the reality that, at any given moment, an agent typically has only **1–5 viable actions** (roads or paths it can take from its current zone). But unlike chess — a one‑on‑one game — these simulations involve a **team game dynamic**: **five seekers** are tasked with locating **one hider**.

This team aspect changes the math of the search entirely. MCTS now needs to consider the **joint actions** of all seekers at each step. For example:  
- If **Agent A** takes Road X and **Agent B** takes Road Y, that’s one unique branch in the tree.  
- If **Agent A** instead takes Road Z while **Agent B** still takes Road Y, that’s a *different* branch.  

Every distinct combination of actions must be accounted for — and the tree expands accordingly.

We’ll also need to **modify the available agent actions** to account for **travel time** and, eventually, **investigation time**. Let’s assume each “turn” in the game represents **one minute of real‑world time**. If an agent commits to a road, it’s locked into that choice for the road’s entire duration. By assuming a travel speed and using each edge’s length, we can calculate this time cost precisely.  

This means we dynamically adjust the **“legal moves”** available to an agent depending on its current state — whether it’s *traveling* or *at rest*. The same principle applies to **investigations**: if an agent chooses to investigate a node, it incurs a time penalty in exchange for the chance of discovering the hider.  

Beyond these modifications, the **MCTS framework fits our abduction simulation naturally.** Because each agent typically has only a handful of viable actions at any give

.

.



## Value Function V(s)

In the AlphaZero framework, the **value function** is defined as a scalar mapping from a game state to an estimate of the long‑term outcome from that state. Formally:

```math
V(θ,s) \approx \mathbb{E}[z \mid s]
```

Where:


| Symbol            | Definition                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `V_(θ,s)`     | The **scalar value function**, parameterized by model weights `θ`, estimating the expected outcome of state `s`. |
| `s`               | The **state** (e.g., board configuration, positions of all agents).        |
| `z`               | The **final game outcome** from the perspective of the current player (e.g., +1 for win, 0 for draw, −1 for loss). |
| `θ`               | The **model parameters/weights** of the neural network producing the value. |

.


The global state vector, **s**,  is defined as follows-

| Feature         | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `t`             | Current timestep.                                                           |
| `H_θ`           | Hider’s **true heading** (absolute, relative to map center, normalized).    |
| `d_Hav (H,i)`   | Haversine distance between seeker `i` and hider `H`.                        |
| `d_Road (H,i)`  | Road distance between seeker `i` and hider `H`.                             |
| `α_(i,θ)`       | Current heading of agent `i` (absolute, relative to map center, normalized).|
| `β_θ`           | Heading to known victim location (absolute, relative to map center, normalized). |
| `ω_i`           | Binary indicator: is agent `i` on a high‑priority node (e.g., hiding spot)? |
| `ω_H`           | Binary indicator: is hider `H` on a high‑priority node (e.g., hiding spot)? |
| `δ`             | Outcome of the simulation (only known after the fact).                      |
| `δ_i`           | Current **status** of agent `i` (e.g., `traveling` or `at rest`).           |
| `t_travel`      | Remaining **travel lock time** for an agent that has committed to a road.   |


.


The neural network we use here is deliberately simple — framed as a regression model that takes the state vector 𝑠 and outputs a scalar value within (-1, 1).

- A value of -1 indicates a state that ultimately led to the hider evading capture.

- A value of +1 indicates a state that ultimately resulted in the hider being captured.

.


### Example network design
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/51b56b51-8b32-4930-94d7-31ae2ac0cc31" />


Later on, inputting the environment itself as a gridded map may be a valuable modification to this function. AlphaZero used multiple convolutional networks to analyze the gridded boards of either Chess or Go, where precense of pieces on a grid were represented by 1s or 0s for a given vector. We avoid this right now, since given that the victory conditions for us, a seeker finding a hider, is far less complex than the movement and capture dynamics of chess. A "blind" analysis of our enivornment should be sufficient.  


---

### Learning Objective

The value head is trained to match the **final game outcome** from self‑play, **with an added proximity bonus** to encourage seekers to move closer to the hider (and penalize the hider for being approached).  

We modify the objective to incorporate this additional shaping term:

```math
\mathcal{L}_v = (V_\theta(s) - (z + \lambda p(s)))^2
```

Where:

| Symbol        | Definition                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `𝓛_v`         | The **loss function** for the value head.                                  |
| `V_θ(s)`      | Predicted scalar value for state `s`.                                      |
| `z`           | Final game result used as the training label (e.g., +1, 0, or −1).         |
| `p(s)`        | **Proximity bonus function**: a shaped reward based on how close seekers are to the hider (positive for seekers, negative for the hider). |
| `λ`           | Weighting coefficient controlling the influence of the proximity bonus on learning. |

The **proximity bonus** provides denser feedback:  
- **Seekers** earn a small positive shaping reward as they close distance to the hider.  
- **The hider** is penalized when seekers get too close, incentivizing evasive movement.  

By blending the game outcome `z` with the shaped proximity term `λ p(s)`, the network learns not only from the final win/loss signal but also from meaningful **intermediate progress**. This helps the value function converge faster and supports deeper planning without requiring exhaustive rollouts.
