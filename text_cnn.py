import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_helpers
from data_helpers import load_data
from data_helpers import batch_iter

#super para need to adjust
vocab_size = ?
sequence_length = ?
filter_sizes = [2,3,4]
embedding_size = 4
num_filters=5# each filter_size has the same number of filter

dev_sample_percentage = 0.1
batch_size = 64
num_epochs = 200
evaluate_every = 100 #Evaluate model on dev set after this many steps (default: 100)
checkpoint_every = 100 #Save model after this many steps (default: 100)

input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

def embedding(input_x):
    # Embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="W")
        embedded_chars = tf.nn.embedding_lookup(W, input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        return embedded_chars_expanded

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
    x_train, x_dev, y_train, y_dev = load_data()
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), batch_size, num_epochs)



    cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))
    train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Training loop. For each batch...
    with tf.Session() as sess:
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_acc = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 1.0})
                print ('step %d, training accuracy %g' % (i, train_acc))
                train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

        test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
        print ("test accuracy %g" % test_acc)