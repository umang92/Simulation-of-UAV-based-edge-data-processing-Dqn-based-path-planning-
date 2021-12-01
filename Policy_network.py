import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Policy_network:
    def __init__(self,inputs):
        num_inputs = 3  ## distance to next location, orientation, battery left
        num_actions = 1  ### it will give you speed
        num_hidden = 128
        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        model = keras.Model(inputs=inputs, outputs=[action, critic])




