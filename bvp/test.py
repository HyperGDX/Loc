import keras
import tensorflow as tf

inputs = tf.keras.Input(shape=(11, 20, 20, 1))
conv_2d_layer = tf.keras.layers.Conv2D(16, (5, 5))
outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
print(outputs.shape)
