import numpy as np
from keras import layers, Model
from keras.layers import Embedding, Dense
import tensorflow as tf


class PolicyHeader(tf.keras.Model):
    def __init__(self, num_edges, embed_dim):
        super().__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.agent_proj = Dense(embed_dim)

        self.edge_embeddings = Embedding(num_edges, embed_dim)
        self.static_proj = Dense(embed_dim)

    def call(self, local_obs_batch, valid_edge_ids_batch, edge_static_features):
        if len(local_obs_batch.shape) == 3:
            local_obs_batch = tf.squeeze(local_obs_batch, axis=1)
        #print(local_obs_batch.shape)
        x = self.dense1(local_obs_batch)    
        x = self.dense2(x)                   
        agent_features = self.agent_proj(x)  

        all_probs = []

        batch_size = tf.shape(agent_features)[0]

        for i in range(5):
            ids = valid_edge_ids_batch[i] 

            valid_edge_vecs = self.edge_embeddings(ids)  
            valid_edge_static = tf.gather(edge_static_features, ids)
            #print(valid_edge_vecs)
            static_proj = self.static_proj(valid_edge_static)  

            final_edge_vecs = valid_edge_vecs + static_proj  

            seeker_vec = agent_features[i]  
            seeker_vec = tf.expand_dims(seeker_vec, axis=0)  

            # Dot: (1, embed_dim) * (num_valid, embed_dim) → (num_valid,)
            scores = tf.reduce_sum(seeker_vec * final_edge_vecs, axis=-1)
            #print(scores)
            probs = tf.nn.softmax(scores) 
            #print(probs)
            all_probs.append(probs)

        return all_probs


