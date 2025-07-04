import numpy as np
from keras import layers, Input, Model, ops
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

    def call(self, local_obs, valid_edge_ids, edge_static_features):
        x = self.dense1(local_obs)
        x = self.dense2(x)
        agent_feature = self.agent_proj(x)  # [batch, embed_dim]

        valid_edge_vecs = self.edge_embeddings(valid_edge_ids)  # [batch, num_valid, embed_dim]
        valid_edge_static = tf.gather(edge_static_features, valid_edge_ids)
        static_proj = self.static_proj(valid_edge_static)
        final_edge_vecs = valid_edge_vecs + static_proj

        agent_feature = tf.expand_dims(agent_feature, axis=1)
        scores = tf.reduce_sum(agent_feature * final_edge_vecs, axis=-1)  # [batch, num_valid]
        policy_probs = tf.nn.softmax(scores)
        return policy_probs


