from nemesis.environment import Environment
from nemesis.Agents import Seeker, Hider
from nemesis.MCTS import MCTSNode, MCTSSearch
from nemesis.globalValueHeader import ValueHeader
from nemesis.localPolicyHeader import PolicyHeader
from nemesis.replayBuffer import ReplayBuffer
from nemesis.replay_io import save_replay_parquet
from nemesis.environment import edgeIDHash, nodeToEdgeHash, static_features, nodes, edges
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import random
import tensorflow as tf
from tensorflow import keras
import json

def getUniqueID():
    now = datetime.now()
    new_time = now + timedelta(minutes=random.randint(1, 7))
    time_part = new_time.strftime("%H%M")
    random_part = random.randint(1000, 9999)
    unique_id = f"{time_part}{random_part}"
    return unique_id


pointer_policies = PolicyHeader(len(edges), embed_dim=64)
critic = ValueHeader(num_seekers=5)
dummy_input = tf.random.normal((1, 37))  # Replace with actual input shape
#critic.build((None,27))  # This builds the model
#critic.load_weights('models/ValueFunction_batch5_r2_93.weights.h5')
print(critic.call(dummy_input))
num_episodes = 500

buffer = ReplayBuffer()

for i,episode in enumerate(range(num_episodes)):
    print(f"Starting episode #{i}")
    root_state = Environment(t=0)
    root_node = MCTSNode(root_state, root_state.seekers)
    state = root_state
    starting_pos = [i.Node for i in state.seekers]
    root_node = MCTSNode(state, state.seekers)
    done = False
    trajectory = []
    outcome = 0
    while not done:
        #print(root_state.nodeToEdgeHash)
        search = MCTSSearch(root_node, pointer_policies, critic, static_features, nodeToEdgeHash)
        search.run(num_simulations=1300)
        best_action = search.best_action()
        per_agent_counts = [{} for _ in range(5)]
        per_agent_probs = root_node.get_training_policy(nodeToEdgeHash)

        state_vector = root_node.state.BuildStateVector()
        observation_vectors = [i.BuildObservationVector() for i in root_node.state.seekers]
        seeker_positions = [i.Node for i in root_node.state.seekers]
        hider_position = root_node.state.hider.Node
        #print(observation_vectors[0])
        trajectory.append({
                            'state_vector': state_vector,
                            'observation_vectors' : observation_vectors,
                            'learned_policy': per_agent_probs,  # [p1, p2, p3, p4, p5]
                            'starting_seeker_positions': starting_pos,
                            'seeker_positions': seeker_positions,
                            'hider_position': hider_position,
                            'value': None  # placeholder for now
                        })
        child_node = root_node.children[best_action]
        root_node = child_node
        root_node.parent = None

        #new_state = root_node.state.simulate_joint_action(best_action)
        #root_node = MCTSNode(new_state, new_state.seekers)
        done = search.is_terminal(root_node)
        
    outcome = search.reward(root_node)
    for i in trajectory:
        i['value'] = outcome

    df = pd.DataFrame(trajectory)
    df['state_vector'] = df['state_vector'].apply(lambda x: x.tolist())
    df['observation_vectors'] = df['observation_vectors'].apply(lambda x: [p.astype(np.float32) for p in x])
    df['learned_policy'] = df['learned_policy'].apply(json.dumps)

    id = getUniqueID()
    df.to_parquet(f'replays/{id}_replay_{episode}.parquet', index=False)

    #print("✅ Saved replay to Parquet")


