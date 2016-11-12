import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
from batch_generators import TripleBatchGenerator

summaries_dir = '/tmp/TensorBoard/summaries/triplet_pretrain'

with tf.variable_scope('inputs'):
    X0 = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='X0')
    X1 = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='X1')
    X2 = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='X2')


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

y0 = triplet_cnn(X0, reuse=False)
y1 = triplet_cnn(X1, reuse=True)
y2 = triplet_cnn(X2, reuse=True)

loss = tf.nn.l2_loss(y0 - y1) - tf.nn.l2_loss(y0 - y2)

try:
    os.rmdir(summaries_dir)
except:
    pass

try:
    os.makedirs(summaries_dir)
except:
    pass

train_writer = tf.train.SummaryWriter(summaries_dir + '/train')
generator = TripleBatchGenerator()

num_steps = 10000

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    train_writer.add_graph(tf.get_default_graph())
    train_writer.close()