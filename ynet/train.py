import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
import ynet.data as data
from ynet.model import Model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch-size', 32, 'Number of samples in batch')
tf.app.flags.DEFINE_boolean("interactive", False, 'Run interactive mode')

tf.app.flags.DEFINE_integer('steps', 100, 'Steps per epoch')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs')
tf.app.flags.DEFINE_integer('width', 128, '---')
tf.app.flags.DEFINE_integer('height', 128, '---')
tf.app.flags.DEFINE_integer('min-size', 200, 'Number of epochs')
tf.app.flags.DEFINE_integer('max-size', 800, 'Number of epochs')

tf.app.flags.DEFINE_string('save-dir', './models', 'Save path')
tf.app.flags.DEFINE_string('data-dir', 'data/', 'Path to input data')

tf.app.flags.DEFINE_float('learning-rate', 0.1, 'Initial LR')
tf.app.flags.DEFINE_float('decay-rate', 0.9, 'Decay rate')


def main(_):
    try:
        os.makedirs(FLAGS.save_dir)
    except:
        pass

    print('Load data')

    path = 'data/ynet-wrinkles'
    flist = [os.path.join(path, f) for f in os.listdir(path)]

    traingen = data.WriDatagen(FLAGS, flist)

    # model = Model(FLAGS, '/cpu:0')

    model = Model(FLAGS)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        sess.run(tf.assign(model.lr, FLAGS.learning_rate))

        print('Let the train begin!')
        for epoch in range(FLAGS.epochs):
            sess.run(tf.assign(model.lr, FLAGS.learning_rate))
            FLAGS.learning_rate *= FLAGS.decay_rate

            pbar = tqdm(range(FLAGS.steps))
            for _ in pbar:
                x, y = next(traingen)
                loss, _, = sess.run([model.loss, model.train_op], {model.inputs: x, model.targets: y})
                pbar.set_description("loss: {:.2f}, ".format(loss))


if __name__ == "__main__":
    tf.app.run()



