from nemesis import environment
import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm


class MCTSNode:
    def __init__(self, state, seekers, parent=None):
        self.state = state                # full env state (time, positions, hider, etc.)
        self.seekers = seekers  # [node_id_1, node_id_2, ..., node_id_5]
        self.parent = parent
        self.children = {}  # key: joint_action_tuple → MCTSNode
        self.per_agent_visit_counts = [{} for _ in range(5)]
        self.P = {}  # priors for each joint action: P[(a1,a2,a3,a4,a5)]
        self.N = {}  # visit counts
        self.W = {}  # total value sum
        self.Q = {}  # mean value = W / N
    
        self.is_terminal = False

    def get_training_policy(self, node_to_edges):
        """
        Generates a complete and correctly normalized policy distribution over all
        legal moves for each agent.
    
        Args:
            node_to_edges (dict): The global map from a node ID to its legal edge IDs.

        Returns:
            list[dict]: A list of policy dictionaries, one for each agent.
        """
        per_agent_policies = []

        # For each seeker agent in this node's state
        for i, seeker in enumerate(self.seekers):
            # 1. Get all raw visit counts and the definitive list of legal moves.
            all_agent_counts = self.per_agent_visit_counts[i]
            legal_moves = node_to_edges.get(seeker.Node, [])

            # 2. CRITICAL FIX: Filter the counts to include ONLY legal moves.
            # This prevents illegal/polluted actions from affecting the normalization.
            legal_counts = {move: all_agent_counts.get(move, 0) for move in legal_moves}

            # 3. Sum the visits of ONLY the legal moves. This is the correct normalization factor.
            total_legal_visits = sum(legal_counts.values())

            policy_for_agent = {}
            if total_legal_visits > 0:
                # 4. Normalize using the sum of legal visits to ensure probabilities sum to 1.
                for move, count in legal_counts.items():
                    policy_for_agent[move] = count / total_legal_visits
            else:
                # Fallback to a uniform policy if no legal moves were ever visited.
                num_legal_moves = len(legal_moves)
                if num_legal_moves > 0:
                    prob = 1.0 / num_legal_moves
                    for move in legal_moves:
                        policy_for_agent[move] = prob
        
            per_agent_policies.append(policy_for_agent)
            
        return per_agent_policies
    
    # def update(self, joint_action, V):
    #     self.N[joint_action] += 1
    #     self.W[joint_action] += V
    #     self.Q[joint_action] = self.W[joint_action] / self.N[joint_action]
    def update(self, joint_action, V):
        if joint_action not in self.N:
            self.N[joint_action] = 0
            self.W[joint_action] = 0.0
            self.Q[joint_action] = 0.0
            
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

        def get_legal_actions_for_agent(agent, node_to_edges):
            """Determines the action space for a single agent."""
            if agent.status == 'traveling':
                # This agent cannot choose a new path. Its only "action"
                # is to continue. We use a special value to represent this.
                return [-1]  # Represents "continue_travel"

            elif agent.status == 'at_node':
                # This agent is free to choose any outgoing edge.
                return node_to_edges.get(agent.Node, [])

        # Get priors for each agent (replace with your actual policy network call)
        for i, seeker in enumerate(self.seekers):
            valid_edges = get_legal_actions_for_agent(seeker, node_to_edges)
            valid_edges = tf.constant(valid_edges, dtype=tf.int32)
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
            # action_node_id = str(environment.edgeIDHash[action_edge.numpy()][1])
            # joint_action.append(action_node_id)
            edge_id = int(action_edge.numpy())  # Just use the raw edge ID
            joint_action.append(edge_id)
        
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



class MCTSSearch:
    def __init__(self, root, pointer_policies, critic, edge_static_features, node_to_edges, c_puct=1.0):
        self.root = root
        self.pointer_policies = pointer_policies  # list of policies, one per seeker
        self.critic = critic
        self.edge_static_features = edge_static_features
        self.node_to_edges = node_to_edges
        self.c_puct = c_puct
    
    def run(self, num_simulations, exploration_epsilon=0.25):
        for i,_ in tqdm(enumerate(range(num_simulations))):
            # if i % 100 == 0:
            #     print(f"Starting Simulation #{i}")
            path = []
            node = self.root

            # 1. Traversal (Selection + Forced Expansion)
            while not self.is_terminal(node):
                if not node.children:
                    # This is a leaf node we are about to expand for the FIRST time.
                    # Initialize all per-agent legal action counts to 0 here.
                    for i, seeker in enumerate(node.seekers):
                        valid_edges = self.node_to_edges[seeker.Node]
                        for edge in valid_edges:
                            # Ensure every legal action has a default count of 0.
                            if edge not in node.per_agent_visit_counts[i]:
                                 node.per_agent_visit_counts[i][edge] = 0
                    break # Exit traversal to expand this leaf node

                if random.random() < exploration_epsilon:
                    # This is a previously expanded node, but we are widening it.
                    # We need to ensure all legal actions are initialized here too.
                    # This check is fast and handles all cases.
                    if not any(node.per_agent_visit_counts): # Check if it's uninitialized
                        for i, seeker in enumerate(node.seekers):
                            valid_edges = self.node_to_edges[seeker.Node]
                            for edge in valid_edges:
                                node.per_agent_visit_counts[i][edge] = 0
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
                V = self.critic.call(new_child_node.state.BuildStateVector()).numpy()[0][0]
                #V = 0.0 # Placeholder
                
                # Add the final step to the path for backpropagation
                # The action taken from `node` is the one that created `new_child_node`
                # We can find it as the last key added to the children dict.
                new_action = list(node.children.keys())[-1]
                path.append((node, new_action))

            # 4. Backpropagation
            for node_on_path, action_taken in reversed(path):
                # Update the joint action value statistics
                node_on_path.update(action_taken, V)

                # ALSO, update the per-agent policy visit counts for this node
                for agent_idx, individual_action in enumerate(action_taken):
                    # Get the dictionary of counts for the current agent
                    counts_dict = node_on_path.per_agent_visit_counts[agent_idx]
        
                    # Robustly increment the count for the specific action taken
                    # If the key doesn't exist, .get() returns 0, so the new value becomes 1.
                    counts_dict[individual_action] = counts_dict.get(individual_action, 0) + 1
                        # NEW: update per-agent counts
                    # for i, a_i in enumerate(action):
                    #     if a_i not in n.per_agent_visit_counts[i]:
                    #         n.per_agent_visit_counts[i][a_i] = 0
                    #     n.per_agent_visit_counts[i][a_i] += 1

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

    def is_terminal(self, node):
        return (
            node.state.timestep >= 120
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