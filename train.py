import tensorflow as tf
from operator import itemgetter


class Trainer(object):
    def __init__(self, optimizer,
                 train_inputs, batch_func_train,
                 valid_inputs=None, batch_func_valid=None,
                 metrics_train=(), metrics_valid=(), valid_freq=100, log_name='log.txt'):

        self.batch_func_train = batch_func_train
        self.batch_func_valid = batch_func_valid
        self.inputs = train_inputs
        self.optimizer = optimizer
        metrics_train = metrics_train if isinstance(metrics_train, (list, tuple)) else list(metrics_train.items())
        self.metric_train_names = list(map(itemgetter(0), metrics_train))
        self.metric_train_tensors = list(map(itemgetter(1), metrics_train))

        metrics_valid = metrics_valid if isinstance(metrics_valid, (list, tuple)) else list(metrics_valid.items())
        self.metric_valid_names = list(map(itemgetter(0), metrics_valid))
        self.metric_valid_tensors = list(map(itemgetter(1), metrics_valid))
        
        self.valid_freq = valid_freq
        self.log_name = log_name
        self.valid_inputs = valid_inputs

    def train(self, num_steps, model_file=None):
        sess = tf.Session()
        log = open(self.log_name, 'w')
        for i in range(num_steps):
            batch = self.batch_func_train()
            feed_dict = {self.inputs[i]: batch[i] for i in range(len(self.inputs))}

            sess.run(self.optimizer, feed_dict=feed_dict)

            if i % self.valid_freq == 0:
                print("Step: " + str(i))

                metric_values = sess.run(self.metric_train_tensors, feed_dict)
                
                for i in range(metric_values):
                    log.write(self.metric_train_tensors[i] + ' ' + str(metric_values[i]) + '\n')
                    print('train ' + self.metric_train_tensors[i] + ' ' + str(metric_values[i]))
                log.write("---\n")
                
                if self.valid_inputs is not None:
                    validation = self.batch_func_valid()
                    feed_dict = {self.valid_inputs[i]: validation[i] for i in range(len(self.inputs))}
    
                    log.write("Step: " + str(i) + "\n")
                    
                    metric_values = sess.run(self.metric_valid_tensors, feed_dict)
    
                    for i in range(metric_values):
                        log.write(self.metric_valid_names[i] + ' ' + str(metric_values[i]) + '\n')
                        print('validation ' + self.metric_valid_names[i] + ' ' + str(metric_values[i]))
                    log.write("---\n")