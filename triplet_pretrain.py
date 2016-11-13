import tensorflow as tf
import tensorflow.contrib.layers as layers


def conv_block(X, size, reuse=None, scope=None):
    with tf.variable_scope('conv_block' if scope is None else scope):
        residual = layers.convolution2d(X, size * 2, (1, 1), reuse=reuse, scope='residual_conv')
        residual = tf.nn.avg_pool(residual, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME')
        y = layers.convolution2d(X, size * 2, (3, 3), 1, reuse=reuse, scope='conv1')
        y = layers.convolution2d(y, size * 2, (3, 3), 1, reuse=reuse, scope='conv2')
        y = layers.max_pool2d(y, (3, 3), 2, padding='SAME')
    return y + residual


def triplet_cnn(X, reuse=None, scope=None):
    with tf.variable_scope('triplet_cnn' if scope is None else scope, reuse=reuse):
        with tf.variable_scope('entry_block'):
            X = layers.convolution2d(X, 32, (3, 3), 1, reuse=reuse, scope='conv1')
            X = layers.convolution2d(X, 64, (3, 3), 1, reuse=reuse, scope='conv2')
            X = layers.max_pool2d(X, (2, 2), 2)  # 64, 64

        X = conv_block(X, 64, reuse=reuse, scope='conv_block1')  # 32, 32
        X = conv_block(X, 128, reuse=reuse, scope='conv_block2')  # 16, 16
        X = conv_block(X, 256, reuse=reuse, scope='conv_block3')  # 8, 8
        X = conv_block(X, 512, reuse=reuse, scope='conv_block4')  # 4, 4
        X = tf.reduce_mean(X, [1, 2])
        X = layers.fully_connected(X, 256, reuse=reuse, scope='fully_connected')
        X = layers.fully_connected(X, 128, reuse=reuse, scope='output', activation_fn=tf.identity)
        return X