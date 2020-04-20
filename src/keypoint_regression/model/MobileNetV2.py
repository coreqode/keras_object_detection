import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, BatchNormalization, Flatten, GlobalAveragePooling2D

class MobileNetV2():
    def __init__(self, config):
        self.input_shape = (config.input_shape, config.input_shape, 3)

    def build_model(self):
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=self.input_shape, include_top=False, weights=None)
        output = GlobalAveragePooling2D()(base_model.output)
#         output = Conv2D(64, 3, padding = 'same')(base_model.output)
#         output = BatchNormalization()(output)
#         output = Activation("relu")(output)
        # output = Conv2D(256, 3, padding = 'same')(output)
        # output = BatchNormalization()(output)
        # output = Activation("relu")(output)
        points = Dense(50,name="box", activation = 'relu')(output)

        model = tf.keras.models.Model(base_model.input, points)
        return model



