import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_helpers
from data_helpers import load_data
import nn
from nn import text_cnn
from data_helpers import batch_iter

# super para need to adjust
filter_sizes = [3, 4, 5]
embedding_size = 4
num_filters = 5  # each filter_size has the same number of filter

dev_sample_percentage = 0.1
batch_size = 64
num_epochs = 200
embedding_dim = 128
evaluate_every = 100  # Evaluate model on dev set after this many steps (default: 100)
checkpoint_every = 100  # Save model after this many steps (default: 100)
dkp = 0.5

x_train, x_dev, y_train, y_dev, vocab_size = load_data()
sequence_length = x_train.shape[1]
num_classes = y_train.shape[1]

with tf.name_scope("feed"):
    input_x = tf.placeholder(tf.int32, [None, sequence_length])
    input_y = tf.placeholder(tf.float32, [None, num_classes])
    dropout_keep_prob = tf.placeholder(tf.float32)

h_dropout, loss, accuracy = text_cnn(
    input_x,
    input_y,
    dropout_keep_prob,
    sequence_length=x_train.shape[1],
    num_classes=y_train.shape[1],
    vocab_size=vocab_size,
    embedding_size=embedding_dim,
    filter_sizes=filter_sizes,
    num_filters=num_filters)

global_step = tf.Variable(0, name="global_step", trainable=False)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)
# Training loop. For each batch...
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        batches = data_helpers.batch_iter(x_train, y_train, batch_size, num_epochs)
        for batch in batches:
            x_batch, y_batch = zip(*batch)  # reverse zip
            _, step_, summaries_, loss_, accuracy_ = sess.run(
                [train_op, global_step, train_op, loss, accuracy],
                feed_dict={input_x: x_batch, input_y: y_batch, dropout_keep_prob: dkp})
            #print "accuracy:", accuracy_
        if i % 1 == 0:
            print "i=", i, "accuracy:", accuracy_