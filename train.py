import tensorflow as tf
from operator import itemgetter
from time import time

class Trainer(object):
    def __init__(self, optimizer,
                 train_inputs, batch_func_train,
                 valid_inputs=None, batch_func_valid=None,
                 metrics_train=(), metrics_valid=(), valid_freq=100, save_freq=2000, log_name='log.txt', save_name=None):

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
        self.save_freq = save_freq
        self.save_name = save_name

    def train(self, num_steps, model_file=None):
        print("Train started for %d steps" % num_steps)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        if self.save_name is not None:
            try:
                saver = tf.train.Saver()
                saver.restore(sess, "models/%s.ckpt" % self.save_name)
            except:
                pass
        log = open(self.log_name, 'w')
        for i in range(num_steps):

            t1 = time()
            batch = self.batch_func_train()
            t2 = time()
            print('batch generation time:', t2 - t1)

            feed_dict = {self.inputs[i]: batch[i] for i in range(len(self.inputs))}

            t1 = time()
            sess.run(self.optimizer, feed_dict=feed_dict)
            t2 = time()
            print('gradient step time:', t2 - t1)

            if i % self.valid_freq == 0:
                print("Step: " + str(i))

                metric_values = sess.run(self.metric_train_tensors, feed_dict)
                
                for i in range(len(metric_values)):
                    log.write(self.metric_train_names[i] + ' ' + str(metric_values[i]) + '\n')
                    print('train ' + self.metric_train_names[i] + ' ' + str(metric_values[i]))
                log.write("---\n")
                
                if self.valid_inputs is not None:
                    validation = self.batch_func_valid()
                    feed_dict = {self.valid_inputs[i]: validation[i] for i in range(len(self.inputs))}
    
                    log.write("Step: " + str(i) + "\n")
                    
                    metric_values = sess.run(self.metric_valid_tensors, feed_dict)
    
                    for i in range(len(metric_values)):
                        log.write(self.metric_valid_names[i] + ' ' + str(metric_values[i]) + '\n')
                        print('validation ' + self.metric_valid_names[i] + ' ' + str(metric_values[i]))
                    log.write("---\n")

            if self.save_name is not None and i % self.save_freq == self.save_freq - 1:
                saver = tf.train.Saver()
                saver.save(sess, "models/%s_%d.ckpt" % (self.save_name, i))
