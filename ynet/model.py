import tensorflow as tf


class Model:
    def __init__(self, opts, gpu='/gpu:0'):
        with tf.device(gpu):
            self.inputs = tf.placeholder(tf.float32, [opts.batch_size, opts.width, opts.height, 3], name='inputs')
            self.targets = tf.placeholder(tf.float32, [opts.batch_size, opts.width, opts.height], name='targets')

            out = self.inputs

            # downsample cascade
            shortcuts = []
            for i, fs in enumerate([32, 64, 128, 256]):
                with tf.name_scope('down_cascade_{}'.format(i)):
                    conved = tf.contrib.layers.convolution2d(out, fs, [3, 3], [1, 1], 'SAME')
                    bn = tf.contrib.layers.batch_norm(conved)
                    out = tf.nn.relu(bn)
                    conved = tf.contrib.layers.convolution2d(out, fs, [3, 3], [1, 1], 'SAME')
                    bn = tf.contrib.layers.batch_norm(conved)
                    out = tf.nn.relu(bn)
                    shortcuts.append(out)
                    out = tf.contrib.layers.max_pool2d(out, [2, 2], [2, 2], 'VALID')
                #
                # print(out)
                # print(out.get_shape()[1:])

            conved = tf.contrib.layers.convolution2d(out, 512, [3, 3], [1, 1], 'SAME')
            bn = tf.contrib.layers.batch_norm(conved)
            out = tf.nn.relu(bn)
            conved = tf.contrib.layers.convolution2d(out, 512, [3, 3], [1, 1], 'SAME')
            bn = tf.contrib.layers.batch_norm(conved)
            out = tf.nn.relu(bn)


            # upsample cascade
            for i, fs in enumerate([256, 128, 64, 32]):
                # this is upsample as it is in tf
                _, w, h, _ = out.get_shape().as_list()
                upscaled = tf.image.resize_bilinear(out, [2 * w, 2 * h])

                # take shortcut
                shortcut = shortcuts.pop()
                merged = tf.concat(3, [upscaled, shortcut], 'merge')

                # just convolve it all
                conved = tf.contrib.layers.convolution2d(merged, fs, [3, 3], [1, 1], 'SAME')
                bn = tf.contrib.layers.batch_norm(conved)
                out = tf.nn.relu(bn)
                conved = tf.contrib.layers.convolution2d(out, fs, [3, 3], [1, 1], 'SAME')
                bn = tf.contrib.layers.batch_norm(conved)
                out = tf.nn.relu(bn)

            logits = tf.contrib.layers.convolution2d(out, 1, [1, 1], [1, 1], 'SAME')
            self.probs = tf.nn.sigmoid(logits)

            costs = tf.nn.sigmoid_cross_entropy_with_logits(tf.squeeze(logits), self.targets)

            cost = tf.reduce_mean(costs)

            self.loss = cost
            self.lr = tf.Variable(0.0, trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss, )

        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(tf.trainable_variables())

    def restore(self, path, session):
        tf.initialize_all_variables().run(session=session)
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(session, ckpt.model_checkpoint_path)


if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(batch_size=7, width=128, height=128)
    model = Model(args)