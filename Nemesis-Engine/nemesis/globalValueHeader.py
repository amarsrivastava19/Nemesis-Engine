import numpy as np
from keras import layers, Input, Model, ops
from keras.layers import Embedding, Dense
import tensorflow as tf


class ValueHeader(tf.keras.Model):
    def __init__(self, num_seekers = 5):
        super().__init__()
        input_dim = 7 + num_seekers * 4  # example feature size
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.output_layer = layers.Dense(1, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        value = self.output_layer(x)
        return value

