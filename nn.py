import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_helpers
from data_helpers import load_data
from data_helpers import batch_iter


def text_cnn(input_x, input_y, dropout_keep_prob, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):

    # with tf.name_scope("feed"):
    #     input_x = tf.placeholder([None, sequence_length], tf.int32)
    #     input_y = tf.placeholder([None, num_classes], tf.float32)
    #     dropout_keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("Embedding_layer"):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            embedded_chars = tf.nn.embedding_lookup(W, input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    with tf.name_scope("dropout"):
        h_dropout = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    with tf.name_scope("output"):
        W = tf.get_variable("W",
                            shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        scores = tf.nn.xw_plus_b(h_dropout, W, b, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions")

    with tf.name_scope("loss"):
        #y_predict = tf.nn.softmax(scores)
        #loss = tf.reduce_sum(input_y*tf.log(y_predict)
        loss = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)

    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    return h_dropout, loss, accuracy
