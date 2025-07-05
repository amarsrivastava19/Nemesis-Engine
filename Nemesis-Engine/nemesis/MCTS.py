from nemesis import environment
import numpy as np
import tensorflow as tf
import random



class MCTSNode:
    def __init__(self, state, seekers, parent=None):
        self.state = state                # full env state (time, positions, hider, etc.)
        self.seekers = seekers  # [node_id_1, node_id_2, ..., node_id_5]
        self.parent = parent
    
        self.children = {}  # key: joint_action_tuple → MCTSNode
    
        self.P = {}  # priors for each joint action: P[(a1,a2,a3,a4,a5)]
        self.N = {}  # visit counts
        self.W = {}  # total value sum
        self.Q = {}  # mean value = W / N
    
        self.is_terminal = False
    

    def update(self, joint_action, V):
        self.N[joint_action] += 1
        self.W[joint_action] += V
        self.Q[joint_action] = self.W[joint_action] / self.N[joint_action]
    
    def expand_one_child(self, pointer_policies, edge_static_features, node_to_edges):
        """
        Samples ONE joint action based on policy priors, creates that child node,
        and returns it. This is the core of sparse expansion.
        """
        priors_per_seeker = []
        valid_edge_sets = []
        all_local_obs = []

        # Get priors for each agent (replace with your actual policy network call)
        for i, seeker in enumerate(self.seekers):
            valid_edges = tf.constant(node_to_edges[seeker.Node], dtype=tf.int32)
            local_obs = seeker.BuildObservationVector()
        
            # NOTE: This should be your policy network's output
            # probs = pointer_policies(...)
            n = len(valid_edges)
            probs = np.ones(n) / n if n > 0 else np.array([])
        
            priors_per_seeker.append(probs)
            valid_edge_sets.append(valid_edges)

        # Sample a single joint action based on individual priors
        joint_action = []
        joint_action_priors = []
        for agent_idx in range(len(valid_edge_sets)):
            p = priors_per_seeker[agent_idx]
            if not np.any(p): continue # Skip if no valid actions

            sampled_idx = np.random.choice(len(valid_edge_sets[agent_idx]), p=p)
            action_edge = valid_edge_sets[agent_idx][sampled_idx]
        
            # Store the actual edge ID for simulation
            action_node_id = str(self.state.edgeIDHash[action_edge.numpy()][1])
            joint_action.append(action_node_id)
        
            # Store the prior for the chosen action
            joint_action_priors.append(p[sampled_idx])

        joint_action = tuple(joint_action)

        # If we've already expanded this action (rare, but possible with sampling),
        # just return the existing child.
        if joint_action in self.children:
            return self.children[joint_action]

        # Initialize statistics for this NEW action
        self.P[joint_action] = np.prod(joint_action_priors) # Joint prior is product of individual priors
        self.W[joint_action] = 0.0
        self.N[joint_action] = 0
        self.Q[joint_action] = 0.0
    
        # Create the single new child node
        child_state = self.state.simulate_joint_action(joint_action)
        child_node = MCTSNode(child_state, child_state.seekers, parent=self)
        self.children[joint_action] = child_node
    
        return child_node
    # def expand(self, pointer_policies, edge_static_features, node_to_edges, critic):
    #     priors = []
    #     valid_edge_sets = [] 
    #     all_local_obs = []
    #     all_valid_edges = []
    #     for i, seeker in enumerate(self.seekers):
    #         valid_edges = tf.constant(node_to_edges[seeker.Node], dtype=tf.int32)
    #         local_obs = seeker.BuildObservationVector()  # 9 features
    #         valid_edge_sets.append(valid_edges)
    #         all_local_obs.append(local_obs)
    #         all_valid_edges.append(valid_edges)

    #     priors_per_seeker = []
    #     for valid_edges in valid_edge_sets:
    #         n = len(valid_edges)
    #         priors_per_seeker.append(np.ones(n) / n)
    #     # probs = [i.numpy().flatten() for i in pointer_policies(
    #     #     np.stack(all_local_obs), all_valid_edges, edge_static_features
    #     # )]
    #     #print(probs)

    #     priors = priors_per_seeker

    
    #     joint_action = []
        
    #     for agent_idx in range(len(valid_edge_sets)):
    #         sampled_idx = np.random.choice(len(valid_edge_sets[agent_idx]), p=priors[agent_idx])
    #         # print(priors[agent_idx])
    #         # print(valid_edge_sets[agent_idx])
    #         action = valid_edge_sets[agent_idx][sampled_idx]
    #         #print(action)
    #         joint_action.append(str(self.state.edgeIDHash[action.numpy()][1]))
 
    #         #print(self.state.edgeIDHash[action.numpy()][1])
    
    #     joint_action = tuple(joint_action)

    #     #V = critic.call(self.state.BuildStateVector()).numpy()[0][0]
    #     V = 0
    #     self.P[joint_action] = tuple(priors)
    #     self.W[joint_action] = 0.0
    #     self.N[joint_action] = 0
    #     self.Q[joint_action] = 0.0
    #     child_state = self.state.simulate_joint_action(joint_action)
    #     child_node = MCTSNode(child_state, child_state.seekers, parent=self)
    #     self.children[joint_action] = child_node
    #     #print(self.children[joint_action])
    #     #print(f"Expanded joint_action: {joint_action}")
    #     #print(f"Children so far: {list(self.children.keys())}")
    #     print(self.N)
    #     return child_node, V, joint_action



class MCTSSearch:
    def __init__(self, root, pointer_policies, critic, edge_static_features, node_to_edges, c_puct=1.0):
        self.root = root
        self.pointer_policies = pointer_policies  # list of policies, one per seeker
        self.critic = critic
        self.edge_static_features = edge_static_features
        self.node_to_edges = node_to_edges
        self.c_puct = c_puct
    
    def run(self, num_simulations, exploration_epsilon=0.25):
        for i,_ in enumerate(range(num_simulations)):
            if i % 100 == 0:
                print(f"Starting Simulation #{i}")
            path = []
            node = self.root

            # 1. Traversal (Selection + Forced Expansion)
            while not self.is_terminal(node):
                
                # If a node is a leaf (has no children), we must expand it.
                if not node.children:
                    break # Exit traversal to expand this leaf node

                # Epsilon-greedy exploration:
                # With a small probability, we force expansion even if children exist.
                # This is the key to widening the tree.
                if random.random() < exploration_epsilon:
                    break # Exit traversal to expand this node and widen the tree

                # Otherwise, select the best child using PUCT and continue down
                joint_action = self.select(node)
                path.append((node, joint_action))
                node = node.children[joint_action]
            
            # `node` is now the leaf we chose to expand

            # 2. Expansion & 3. Evaluation
            if self.is_terminal(node):
                V = self.reward(node)
            else:
                # Expand the chosen leaf node by adding ONE new child
                new_child_node = node.expand_one_child(
                    self.pointer_policies,
                    self.edge_static_features,
                    self.node_to_edges
                )
                
                # Evaluate the NEW node with the critic
                # V = self.critic.call(new_child_node.state.BuildStateVector()).numpy()[0][0]
                V = 0.0 # Placeholder
                
                # Add the final step to the path for backpropagation
                # The action taken from `node` is the one that created `new_child_node`
                # We can find it as the last key added to the children dict.
                new_action = list(node.children.keys())[-1]
                path.append((node, new_action))

            # 4. Backpropagation
            for n, action in reversed(path):
                n.update(action, V)
    # def run(self, num_simulations):
    #         for i,_ in enumerate(range(num_simulations)):
    #             #print(f"Starting run {i}")
    #             path = []
    #             node = self.root
    #             joint_action = []
    #             while not self.is_terminal(node):
    #                 for seeker in node.seekers:
    #                     valid_edges = self.node_to_edges[seeker.Node]
    #                     joint_action.append(random.choice(valid_edges))
    #                 joint_action = tuple(joint_action)
    #                 node.children[joint_action] = None
    #                 joint_action = self.select(node)
    #                 path.append((node, joint_action) )
    #                 node = node.children[joint_action]

    #                 # If you expanded:
    #                 if not self.is_terminal(node):
    #                     child, V , joint_action = node.expand(self.pointer_policies,
    #                                             self.edge_static_features,
    #                                             self.node_to_edges,
    #                                             self.critic)
    #                     # Store expansion action too:
    #                     node.children[joint_action] = child
    #                     path.append( (node, joint_action) )
    #                 else:
    #                     V = self.reward(node)

    #             # Now backprop:
    #             for n, action in reversed(path):
    #                 n.update(action, V)

    def select(self, node):
        N_total = sum(node.N.values())
        best_score = -float('inf')
        best_action = None

        # Iterate over the actions that have been initialized for this node
        for action in node.children.keys():
            if action not in node.Q:
                continue # Skip actions that haven't been properly initialized

            Q = node.Q[action]
            P = node.P[action]  # Use the stored joint prior directly
            N_action = node.N[action]
            
            # The UCB1 formula from AlphaZero
            U = Q + self.c_puct * P * np.sqrt(N_total) / (1 + N_action)

            if U > best_score:
                best_score = U
                best_action = action
                
        return best_action
    # def select(self, node):
    #     N_total = sum(node.N.values()) + 1e-8
    #     best_score = -float('inf')
    #     best_action = None

    #     for action in node.children:
    #         #print(action)
    #         Q = node.Q[action]
    #         #print(Q)
    #         P = np.mean([np.mean(p) for p in node.P[action]])  # avg local priors
    #         #print(P)
    #         N_action = node.N[action]
    #         #print(N_action)
    #         #print(V)

    #         U = Q + self.c_puct * P * np.sqrt(N_total) / (1 + N_action)

    #         if U > best_score:
    #             best_score = U
    #             best_action = action

    #     return best_action

    def is_terminal(self, node):
        return (
            node.state.timestep >= 30
            or any(seeker.Node == node.state.hider.Node for seeker in node.state.seekers)
        )

    def reward(self, node):
        if any(seeker.Node == node.state.hider.Node for seeker in node.state.seekers):
            return 1.0
        else:
            return -1.0

    def best_action(self):
        # Pick the joint action with highest visit count.
        visits = [(action, self.root.N[action]) for action in self.root.children]
        best = max(visits, key=lambda x: x[1])
        return best[0]