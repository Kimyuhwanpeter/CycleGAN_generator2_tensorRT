# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math

def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,):
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)
    
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        p = int((3 - 1) / 2)
        
        h = tf.keras.layers.ZeroPadding2D((p,p))(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = tf.keras.layers.BatchNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)
        
        h = tf.keras.layers.ZeroPadding2D((p,p))(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = tf.keras.layers.BatchNormalization(epsilon=1e-5)(h)

        return tf.keras.layers.ReLU()(tf.keras.layers.add([x, h]))

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)
    
    # 1
    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = tf.keras.layers.BatchNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = tf.keras.layers.BatchNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)

    # 5
    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.keras.layers.Activation('tanh')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)