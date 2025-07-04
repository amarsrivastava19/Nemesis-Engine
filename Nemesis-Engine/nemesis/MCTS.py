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
    
        self.is_expanded = False
        self.is_terminal = False
    
    def expand(self, pointer_policies, edge_static_features, node_to_edges, critic):
        priors = []
        valid_edge_sets = [] 
    
        for i, seeker in enumerate(self.seekers):
            valid_edges = node_to_edges[seeker.Node]
            local_obs = seeker.BuildObservationVector()  # 9 features
            probs = pointer_policies[i](
                local_obs, valid_edges, edge_static_features
            ).numpy()
            priors.append(probs)
            valid_edge_sets.append(valid_edges)
    
        joint_action = []
        
        for agent_idx in range(len(valid_edge_sets)):
            sampled_idx = np.random.choice(len(valid_edge_sets[agent_idx]), p=priors[agent_idx])
            action = valid_edge_sets[agent_idx][sampled_idx]
            joint_action.append(action)
    
        joint_action = tuple(joint_action)
        self.P[joint_action] = tuple(priors)  # store full local priors for later
    
        V = critic(self.state).numpy()
        self.W[joint_action] = 0.0
        self.N[joint_action] = 0
        self.Q[joint_action] = 0.0
    
        child_state, child_seekers = simulate_joint_action(self.state, self.seekers, joint_action)
        child_node = MCTSNode(child_state, child_seekers, parent=self)
        self.children[joint_action] = child_node
    
        self.is_expanded = True
        return child_node, V