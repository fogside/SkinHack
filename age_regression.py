import numpy as np
from batch_generators import AgeGenderBatchGeneratorFolder
from train import Trainer
import tensorflow as tf
import tensorflow.contrib.layers as layers

input_size = 512
batch_size = 16
output_size = 50

gen = AgeGenderBatchGeneratorFolder(batch_size, (input_size, input_size),
                                    (int(input_size * 0.8), int(input_size * 0.8)),
                                    (200, 200))

import scipy.ndimage.filters as filters


def conv_block(X, size, reuse=None, scope=None):
    with tf.variable_scope('conv_block' if scope is None else scope):
        residual = layers.convolution2d(X, size * 2, (1, 1), stride=4, reuse=reuse, scope='residual_conv')
        y = layers.convolution2d(X, size * 2, (3, 3), 1, reuse=reuse, scope='conv1')
        y = layers.max_pool2d(y, (3, 3), 2, padding='SAME')
        y = layers.convolution2d(y, size * 2, (3, 3), 1, reuse=reuse, scope='conv2')
        y = layers.max_pool2d(y, (3, 3), 2, padding='SAME')
    return y + residual


def cnn(X, reuse=None, use_dropout=True, scope=None):
    with tf.variable_scope('triplet_cnn' if scope is None else scope, reuse=reuse):
        with tf.variable_scope('entry_block'):
            X = layers.convolution2d(X, 32, (3, 3), 1, reuse=reuse, scope='conv1')
            X = layers.convolution2d(X, 64, (3, 3), 1, reuse=reuse, scope='conv2')
            X = layers.max_pool2d(X, (3, 3), 2)  # 64, 64

        X /= 255.
        X = conv_block(X, 64, reuse=reuse,  scope='conv_block1')
        X = conv_block(X, 128, reuse=reuse, scope='conv_block2')
        X = conv_block(X, 256, reuse=reuse, scope='conv_block3')
        X = tf.reduce_mean(X, [1, 2])
        if use_dropout:
            X = tf.nn.dropout(X, 0.5)
        X = layers.fully_connected(X, 128, reuse=reuse, scope='fully_connected')
        if use_dropout:
            X = tf.nn.dropout(X, 0.7)
        X = layers.fully_connected(X, output_size, reuse=reuse, scope='output', activation_fn=tf.identity)
        return X


with tf.variable_scope('inputs'):
    X_train = tf.placeholder(dtype=tf.float32, shape=(None, input_size, input_size, 3), name='X_train')
    y_train = tf.placeholder(dtype=tf.float32, shape=(None, output_size), name='y_train')

    X_valid = tf.placeholder(dtype=tf.float32, shape=(None, input_size, input_size, 3), name='X_train')
    y_valid = tf.placeholder(dtype=tf.float32, shape=(None, output_size), name='y_train')

train_logits = cnn(X_train)
train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, y_train))
train_mse = tf.reduce_mean(tf.nn.l2_loss(tf.cast(tf.arg_max(train_logits, 1) - tf.argmax(y_train, 1), tf.float32))) ** 0.5

valid_logits = cnn(X_valid, use_dropout=False, reuse=True)
valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, y_valid))
valid_mse = tf.reduce_mean(tf.nn.l2_loss(tf.cast(tf.arg_max(valid_logits, 1) - tf.argmax(y_valid, 1), tf.float32))) ** 0.5

optimizer = tf.train.AdagradOptimizer(0.3).minimize(train_loss)

n_step = 1
def get_batch():
    global n_step
    X, y, _ = gen.get_supervised_batch()
    _y = np.zeros((X.shape[0], output_size), dtype=np.float32)
    _y[np.arange(_y.shape[0]), (y * output_size) // 100] = np.float32(1.)
    _y = filters.gaussian_filter1d(_y, 4. / n_step ** 0.25)
    n_step += 1
    return X, _y


trainer = Trainer(optimizer, [X_train, y_train], get_batch, [X_valid, y_valid], get_batch,
                  metrics_train=[('train_loss', train_loss), ('train_MSE', train_mse)],
                  metrics_valid=[('valid_loss', valid_loss), ('valid_MSE', valid_mse)],
                  log_name='age_regression_cnn.log', save_name='age_regression_cnn_simple')

trainer.train(200000)