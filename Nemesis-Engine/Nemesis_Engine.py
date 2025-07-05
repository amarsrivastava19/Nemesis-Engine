from nemesis.environment import Environment
from nemesis.Agents import Seeker, Hider
from nemesis.MCTS import MCTSNode, MCTSSearch
from nemesis.globalValueHeader import ValueHeader
from nemesis.localPolicyHeader import PolicyHeader
from nemesis.replayBuffer import ReplayBuffer
from nemesis.replay_io import save_replay_parquet
import pandas as pd
import numpy as np
from collections import defaultdict


nodes = pd.read_csv('data/DSMMetro_nodes.csv')
edges = pd.read_csv('data/DSMMetro_edges.csv')

root_state = Environment(nodes,edges)
root_node = MCTSNode(root_state, root_state.seekers)

pointer_policies = PolicyHeader(len(edges), embed_dim=64)
critic = ValueHeader(num_seekers=len(root_state.seekers))


num_episodes = 30

buffer = ReplayBuffer()

for i,episode in enumerate(range(num_episodes)):
    print(f"Starting episode #{i}")
    state = root_state
    root_node = MCTSNode(state, state.seekers)
    done = False
    trajectory = []
    outcome = 0

    while not done:
        #print(root_state.nodeToEdgeHash)
        search = MCTSSearch(root_node, pointer_policies, critic, root_state.static_features, root_state.nodeToEdgeHash)
        search.run(num_simulations=1600)
        best_action = search.best_action()
        visit_counts = np.zeros_like(best_action)
                                     
        joint_counts = root_node.N
        joint_action = root_node.children
        per_agent_counts = [defaultdict(int) for _ in range(5)]

        for action, count in root_node.N.items():
            for i, a_i in enumerate(action):
                per_agent_counts[i][a_i] += count
        per_agent_probs = []

        for seeker_counts in per_agent_counts:
            total = sum(seeker_counts.values())
            probs = {edge: c/total for edge,c in seeker_counts.items()}
            #print(probs.values())
            per_agent_probs.append(probs.values())

        state_vector = root_node.state.BuildStateVector()
        trajectory.append({
                            'state': state_vector,
                            'policy': per_agent_probs,  # [p1, p2, p3, p4, p5]
                            'value': None  # placeholder for now
                        })

        new_state = root_node.state.simulate_joint_action(best_action)
        root_node = MCTSNode(new_state, new_state.seekers)
        done = search.is_terminal(root_node)
        
    outcome = search.reward(root_node)
    for i in trajectory:
        i['value'] = outcome

    df = pd.DataFrame(trajectory)
    df['state'] = df['state'].apply(lambda x: x.tolist())
    df['policy'] = df['policy'].apply(lambda x: [p for p in x])
    df.to_parquet(f'replays/replay_{episode}.parquet', index=False)

    print("✅ Saved replay to Parquet")
    #print(f"Episode {episode} done — buffer size: {len(buffer)}")


