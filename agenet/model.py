import tensorflow as tf


class AgeModel:
    def __init__(self, opts, gpu='/gpu:0'):
        with tf.device(gpu):
            self.inputs = tf.placeholder(tf.float32, [opts.batch_size, opts.width, opts.height, 3], name='inputs')
            self.targets = tf.placeholder(tf.float32, [opts.batch_size, opts.bins], name='targets')

            out = self.inputs

            for fs in [32, 64, 128]:
                conved = tf.contrib.layers.convolution2d(out, fs, [3, 3], [1, 1], 'SAME')
                bn = tf.contrib.layers.batch_norm(conved)
                out = tf.nn.relu(bn)
                conved = tf.contrib.layers.convolution2d(out, fs, [3, 3], [1, 1], 'SAME')
                bn = tf.contrib.layers.batch_norm(conved)
                out = tf.nn.relu(bn)
                out = tf.contrib.layers.max_pool2d(out, [2, 2], [2, 2], 'VALID')

                print(out)
                print(out.get_shape()[1:])

            conved = tf.contrib.layers.convolution2d(out, 128, [3, 3], [1, 1], 'SAME')
            bn = tf.contrib.layers.batch_norm(conved)
            out = tf.nn.relu(bn)

            print(out)
            print(out.get_shape()[1:])

            # Let's make GAP = global average pooling.
            # 1. Take number of channels == wanted output size
            conved = tf.contrib.layers.convolution2d(out, opts.bins, [3, 3], [1, 1], 'SAME')

            print(out)
            print(out.get_shape()[1:])

            # 2. Use appropriate size avg pooling over feature maps
            # _sizes are ad-hocked, so tune if needed
            out = tf.nn.avg_pool(out, ksize=[1, 12, 12, 1], strides=[1, 12, 12, 1], padding='SAME')

            print(out)
            print(out.get_shape()[1:])

            # 3. Now squeeze [bs, 1, 1, bins] to [bs, bins] and CE with prior distribution :)
            lpde = tf.nn.log_softmax(tf.squeeze(out))
            self.lpde = lpde

            costs = - tf.reduce_sum(lpde * self.targets, reduction_indices=1)
            cost = tf.reduce_mean(costs)

            self.loss = tf.reduce_mean(cost)
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
    args = Namespace(batch_size=7, width=100, height=100, bins=53)
    model = AgeModel(args)