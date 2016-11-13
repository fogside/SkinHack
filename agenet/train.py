import os

import tensorflow as tf
from tqdm import tqdm

import data
import utils
from agenet.model import AgeModel

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch-size', 32, 'Number of samples in batch')
tf.app.flags.DEFINE_boolean("interactive", False, 'Run interactive mode')

tf.app.flags.DEFINE_integer('steps', 100, 'Steps per epoch')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs')
tf.app.flags.DEFINE_integer('width', 100, 'Number of epochs')
tf.app.flags.DEFINE_integer('min-size', 200, 'Number of epochs')
tf.app.flags.DEFINE_integer('max-size', 800, 'Number of epochs')
tf.app.flags.DEFINE_integer('height', 100, 'Number of epochs')
tf.app.flags.DEFINE_integer('bins', 53, 'Number of epochs')

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
    full_data = utils.load_age_data(FLAGS.data_dir)

    train_data, val_data = utils.split_data(full_data)
    traingen, valgen = data.AgeDatagen(FLAGS, train_data), data.AgeDatagen(FLAGS, val_data)
    # model = Model(FLAGS, '/cpu:0')

    model = AgeModel(FLAGS)

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



