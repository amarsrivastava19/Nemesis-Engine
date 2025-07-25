import numpy as np
from keras import layers, Input, Model, ops
from keras.layers import Embedding, Dense
import tensorflow as tf

class ValueHeader(tf.keras.Model):
    def __init__(self, num_seekers=5):
        super().__init__()
        input_dim = 7 + num_seekers * 6
        
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        
        self.dense3 = layers.Dense(256, activation='relu')
        self.dropout3 = layers.Dropout(0.2)
        
        self.dense4 = layers.Dense(256, activation='relu')
        self.dropout4 = layers.Dropout(0.2)
        
        self.output_layer = layers.Dense(1, activation='tanh')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        x = self.dense4(x)
        x = self.dropout4(x, training=training)
        value = self.output_layer(x)
        return value
