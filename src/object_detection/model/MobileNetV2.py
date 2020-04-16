import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

class MobileNetV2():
    def __init__(self, config):
        self.input_shape = (config.input_shape, config.input_shape, 3)
        if config.inference:
            self.weights = None
        else:
            self.weights = 'imagenet'

    def build_model(self):
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=self.input_shape, alpha=1.0, include_top=True, weights=self.weights, input_tensor=None, pooling=None, classes=1000)


        loc_layer_dense = Dense(8, activation = 'relu')(base_model.output)
        conf_layer_dense = Dense(2, activation = 'sigmoid')(base_model.output)

        output = tf.keras.layers.concatenate([conf_layer_dense, loc_layer_dense], axis = -1)
        model = tf.keras.models.Model(base_model.input, output)
        return model



    