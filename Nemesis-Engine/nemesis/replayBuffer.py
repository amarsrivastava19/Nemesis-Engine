class ReplayBuffer:
    def __init__(self):
        self.data = []

    def store(self, state_vector, mcts_policy, outcome):
        self.data.append({
            'state': state_vector,
            'policy': mcts_policy,
            'value': outcome
        })

    def sample(self, batch_size):
        idx = np.random.choice(len(self.data), size=batch_size, replace=False)
        batch = [self.data[i] for i in idx]
        states = np.stack([b['state'] for b in batch])
        policies = np.stack([b['policy'] for b in batch])
        values = np.stack([b['value'] for b in batch])
        return states, policies, values

    def __len__(self):
        return len(self.data)