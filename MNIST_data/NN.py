#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/13 11:34
# @Author  : GL
# @File    : NN.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
TRAIN_STEP = 10000
LEARNING_RATE = 0.01
L2NORM_RATE = 0.001


def train(mnist):
    X = tf.placeholder(tf.float32, shape=[None, INPUT_NODE])
    y = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE])

    w1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    w2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    layer1 = tf.nn.relu(tf.nn.xw_plus_b(X, w1, b1))
    y_hat = tf.nn.xw_plus_b(layer1, w2, b2)
    print("1 step is ok")
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2)
    loss = cross_entropy_mean + L2NORM_RATE * regularization
    print("2 step is ok")
    accuracy = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy_ratio = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    print("3 step is ok")
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
    print("4 step is ok")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("5 step is ok")
        for i in range(TRAIN_STEP):
            Xs, ys = mnist.train.next_batch(BATCH_SIZE)
            feed_dict = {
                X: Xs,
                y: ys
            }
            _, step, train_acc, train_loss = sess.run([train_op, global_step, accuracy_ratio, loss],
                                                      feed_dict=feed_dict)
            if (i % 100 == 0):
                print("After %d steps, in train data, loss is %g, accuracy is %g. " % (step, train_loss, train_acc))

        test_feed = {
            X: mnist.test.images,
            y: mnist.test.labels
        }
        test_acc = sess.run(accuracy_ratio, feed_dict=test_feed)
        print("After %d steos, in test dataset, accuracy is %g" % (TRAIN_STEP, test_acc))


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    print(mnist.train.images.shape, mnist.train.labels.shape)
    train(mnist)
