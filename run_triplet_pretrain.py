import tensorflow as tf
from batch_generators import TripleBatchGeneratorFolder, ImageFolderReader
from triplet_pretrain import triplet_cnn
from train import Trainer
import os
import numpy as np

print('Loading data...')
reader = ImageFolderReader('data/ddp')
generator = TripleBatchGeneratorFolder("data/ddp", 100, (128, 128), (100, 100), (70, 70))
print('Done!')

summaries_dir = '/tmp/TensorBoard/summaries/triplet_pretrain'

with tf.variable_scope('inputs'):
    X0 = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='X0')
    X1 = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='X1')
    X2 = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='X2')

y0 = triplet_cnn(X0, reuse=False)
y1 = triplet_cnn(X1, reuse=True)
y2 = triplet_cnn(X2, reuse=True)

loss = tf.nn.l2_loss(y0 - y1) - tf.nn.l2_loss(y0 - y2)

optimizer = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

try:
    os.rmdir(summaries_dir)
except:
    pass

try:
    os.makedirs(summaries_dir)
except:
    pass

#train_writer = tf.train.SummaryWriter(summaries_dir + '/train')

num_steps = 10000
"""
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    train_writer.add_graph(tf.get_default_graph())

    train_writer.close()
"""
trainer = Trainer(optimizer, (X0, X1, X2), generator.get_triple_batch, metrics_train=(('loss', loss),))
trainer.train(num_steps)