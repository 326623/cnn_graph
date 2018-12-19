import tensorflow as tf
from tqdm import tqdm
import sklearn
import numpy as np

class BaseModel(object):
    def __init__(self):
        self._regularizers = []

    def predict_with_labels(self, data, labels, sess=None):
        """
        Args:
          data: `tf.data.dataset` containing input features
          labels: `tf.data.dataset` containing input labels
          sess: the current `tf.Session` or None

        Returns:
          batch_labels: `numpy` ...
          batch_loss: `numpy` ...
          batch_prediction: `numpy` ...
        """
        data_init_op = self.data_iterator.make_initializer(data)
        labels_init_op = self.labels_iterator.make_initializer(labels)
        feed_dict = {self.ph_dropout: 1.0}
        sess = self._get_session(sess)

        try:
            sess.run([data_init_op, labels_init_op])
            while True:
                batch_labels, batch_loss, batch_predictions = sess.run(
                    [self.pipe_labels, self.op_loss, self.op_prediction], feed_dict)
                # In the event of large dataset, we may want to write prediction
                # to save memory, the limitation is the lack of multi-thread support
                # with the keyword yield
                yield batch_labels, batch_loss, batch_predictions
        except tf.errors.OutOfRangeError:
            pass

    def predict(self, data, sess=None):
        """
        Args:
          data: `tf.data.dataset` containing input features
          labels: `tf.data.dataset` containing input labels
          sess: the current `tf.Session` or None

        Returns:
          batch_labels: `numpy` ...
          batch_loss: `numpy` ...
          batch_prediction: `numpy` ...
        """
        data_init_op = self.data_iterator.make_initializer(data)
        feed_dict = {self.ph_dropout: 1.0}
        sess = self._get_session(sess)

        try:
            sess.run([data_init_op])
            while True:
                batch_predictions = sess.run(
                    [self.op_logits, self.op_prediction], feed_dict)
                # In the event of large dataset, we may want to write prediction
                # to save memory
                yield batch_predictions
        except tf.errors.OutOfRangeError:
            return

    def evaluate(self, data, labels, total, sess=None):

        # collected labels, prediction, loss
        c_labels, c_predictions = np.empty(total, dtype=int), np.empty(total, dtype=int)
        loss = 0.0
        begin = 0
        end = 0
        for batch_labels, batch_loss, batch_predictions \
            in self.predict_with_labels(data, labels, sess):
            # batch size
            end += len(batch_labels)
            # collecting
            c_labels[begin:end] = batch_labels[:]
            c_predictions[begin:end] = batch_predictions[:]
            loss += batch_loss
            begin = end
        accuracy = 100 * sklearn.metrics.accuracy_score(c_labels, c_predictions)
        f1 = 100 * sklearn.metrics.f1_score(c_labels, c_predictions, average='weighted')
        return accuracy, f1, loss / total

    def inference(self, data, dropout):
        """
        This method that build the main graph
        """
        logits = self._inference(data, dropout)
        return logits

    def fit(self, training_data, training_labels, val_data, val_labels,
            train_total, val_total, batch_size, num_epochs,
            progress_per=100, eval_every_n_epochs=None):
        """
        Runs the entire epoch of the train_dataset
        training_data:
        training_labels:
        val_data:
        val_labels:
        progress_per:
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=self.graph, config=config)
        # some model saving here
        sess.run(self.op_init)
        training_data_init_op = self.data_iterator.make_initializer(training_data)
        training_labels_init_op = self.labels_iterator.make_initializer(training_labels)
        sess.run([training_data_init_op, training_labels_init_op])

        val_accuracy, val_loss = -float('inf'), float('inf')
        pbar = tqdm(total=train_total*num_epochs/batch_size, mininterval=1)
        for n_epochs in range(num_epochs):
            sess.run([training_data_init_op, training_labels_init_op])
            try:
                while True:
                    feed_dict = {self.ph_dropout: self.dropout}
                    # The only guarantee tensorflow makes about order of execution is that
                    # all dependencies (either data or control) of an op are executed
                    # before that op gets executed.
                    global_step, loss, loss_average, train_op = \
                        sess.run([self.global_step, self.op_loss, self.op_loss_average, self.op_train], feed_dict)
                    pbar.update()
                    pbar.set_postfix(trn_loss="{:.3e}".format(loss_average),
                                     val_accuracy="{:3.3f}".format(val_accuracy),
                                     val_loss="{:.3e}".format(val_loss),
                                     refresh=False)
                    # tf.logging.log_every_n(tf.logging.INFO,
                    #                        'step: %d, trn_loss: %.3e, val_accuracy: %.3e, val_loss: %.3e',
                    #                        progress_per, global_step, loss, val_accuracy, val_loss)
                    # if total:
                    #     tf.logging.log_every_n(tf.logging.INFO, 'step: %d', progress_per, global_step)
                    # tf.logging.log_every_n(tf.logging.INFO, 'loss: %f, average loss: %f',
                    #                        progress_per, loss, loss_average)

            except tf.errors.OutOfRangeError:
                pass

            if eval_every_n_epochs is not None and \
               n_epochs % eval_every_n_epochs == 0 \
               and n_epochs != num_epochs-1: # not the last epochs, since at the end already validated
                val_accuracy, val_f1, val_loss = \
                    self.evaluate(val_data, val_labels, val_total, sess)

        # evaluation at the end
        self.evaluate(val_data, val_labels, val_total, sess)
        tf.logging.info('val_accuracy: %f, val_f1: %f, val_loss: %f',
                            val_accuracy, val_f1, val_loss)
        pbar.close()
        return sess

    def training(self, loss, learning_rate, decay_steps=55000/200, decay_rate=0.95, momentum=0.9):
        with tf.name_scope('training'):
            # learning rate
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, self.global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            if momentum == 0:
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, self.global_step)
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    def probabilties(self, logits):
        """
        Returns the probability of a sample to belong to each class.
        """
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization=None):
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)

            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self._regularizers)
                loss = cross_entropy + regularization

            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                # tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                # tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                # tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')

            return loss, loss_average

    #def input_pipeline(self, data_format, labels_format):
    def build_graph(self, data_format, labels_format,
                    decay_steps, decay_rate, momentum, graph=None):
        """
        data_format, labels_format: output_types, output_shapes of data_dataset, labels_dataset
        distinguish data and labels because some operations don't depend on labels, e.g.,
        predict, inference etc.
        """
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.momentum = momentum
        if graph:
            self.graph = graph
        else:
            self.graph = tf.Graph()

        with self.graph.as_default():

            # Inputs
            with tf.name_scope('inputs'):
                self.data_iterator = tf.data.Iterator.from_structure(*data_format)
                self.labels_iterator = tf.data.Iterator.from_structure(*labels_format)
                self.pipe_data = self.data_iterator.get_next()
                self.pipe_labels = self.labels_iterator.get_next()
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                #self.pipe_data, self.pipe_labels, self.pipe_dropout \
                    #= input_iterator.get_next()

            self.op_logits = self.inference(self.pipe_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(self.op_logits,
                                                           self.pipe_labels,
                                                           self.regularization)
            # self.op_loss = self.loss(self.op_logits, self.pipe_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,)
                                          #self.decay_steps, self.decay_rate,
                                          #self.momentum)
            self.op_prediction = self.prediction(self.op_logits)
            self.op_init = tf.global_variables_initializer()

            # self.op_summary = tf.summary.merge_all()
        #     self.op_saver = tf.train.Saver(max_to_keep=5)

        # self.graph.finalize()

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self._regularizers.append(tf.nn.l2_loss(var))
            # tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self._regularizers.append(tf.nn.l2_loss(var))
            # tf.summary.histogram(var.op.name, var)
        return var

    def _get_session(self, sess=None):
        if sess is None:
            sess = tf.Session(graph=self.graph)
        return sess
