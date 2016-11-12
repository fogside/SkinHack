import tensorflow as tf
from operator import itemgetter


class Trainer(object):
    def __init__(self, num_steps, optimizer,
                 train_inputs, batch_func_train,
                 valid_inputs=None, batch_func_valid=None,
                 metrics=(), valid_freq=100, log_name='log.txt', model_file=None):

        self.batch_func_train = batch_func_train
        self.batch_func_valid = batch_func_valid
        self.inputs = train_inputs
        self.optimizer = optimizer
        metrics = metrics if isinstance(metrics, (list, tuple)) else list(metrics.items())
        self.metric_names = list(map(itemgetter(0), metrics))
        self.metric_tensors = list(map(itemgetter(1), metrics))
        self.num_steps = num_steps
        self.valid_freq = valid_freq
        self.log_name = log_name
        self.valid_inputs = valid_inputs
        self.model_file=None

    def train(self):
        sess = tf.Session()
        log = open(self.log_name, 'w')
        for i in range(self.num_steps):
            batch = self.batch_func_train()
            feed_dict = {self.inputs[i]: batch[i] for i in range(len(self.inputs))}

            sess.run(self.optimizer, feed_dict=feed_dict)

            if i % self.valid_freq == 0 and self.valid_inputs is not None:
                validation = self.batch_func_valid()
                feed_dict = {self.valid_inputs[i]: validation[i] for i in range(len(self.inputs))}

                log.write("Step: " + str(i) + "\n")
                metric_values = sess.run(self.metric_tensors, feed_dict)

                for i in range(metric_values):
                    log.write(self.metric_names[i] + ' ' + str(metric_values[i]) + '\n')
                log.write("---\n")