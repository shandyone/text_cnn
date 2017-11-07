import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filter_sizes = [2,3,4]
embedding_size = 4
num_filters=5# each filter_size has the same number of filter

def embedding(input_x):
    # Embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="W")
        embedded_chars = tf.nn.embedding_lookup(W, input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        return

def conv(embedded_chars_expanded):
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h,ksize=[1, sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID',name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(3, pooled_outputs)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    return h_pool_flat

def dropout(x):
    return tf.nn.dropout(x,keep_prob=0.5)

def training():
