# coding: utf-8
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import config
from data import Data, batch_iter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # for issue: The TensorFlow library wasn't compiled to use SSE3


class Model(object):
    def __init__(self, we):
        self.seq_len = 120
        self.embed_size = 100
        self.class_num = 18
        self.lstm_size = 100
        self.grad_clip = 10
        self.learning_rate = tf.placeholder(tf.float32)
        tf.set_random_seed(1234)

        self.in_x = tf.placeholder(tf.int32, [None, None])  # shape: (batch x seq)
        self.in_y = tf.placeholder(tf.int32, [None])
        self.in_len = tf.placeholder(tf.int32, [None])
        self.y_prob, self.y_p, self.cost, self.train_op, self.acc_cnt = self._build_model(we)
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self, we):
        # Embedding layer
        self.we = tf.Variable(we)
        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_x)   # shape: (batch x time_step x word_dim)
        # Create a lstm layer
        cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_size)
        #  stack lstm : tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)
        # Get layer activations (second output is the final state of the layer, do not need)
        # [batch, time_step, n_hidden]
        cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_size)
        b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedded_seq,
                                                              self.in_len, dtype=tf.float32)

        contex = tf.concat([b_outputs[0], embedded_seq, b_outputs[1]], axis=2)
        contex = tf.expand_dims(contex, -1)
        pooled = tf.nn.max_pool(contex,
                                ksize=[1, self.seq_len, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name="pool")
        pooled = tf.squeeze(pooled)

        w_h = tf.Variable(tf.random_normal([self.lstm_size * 2 + self.embed_size, 50], stddev=0.01))
        b_h = tf.Variable(tf.zeros([50]))
        hidden = tf.nn.xw_plus_b(pooled, w_h, b_h)
        w = tf.Variable(tf.random_normal([50, self.class_num], stddev=0.01))
        b = tf.Variable(tf.zeros([self.class_num]))
        logits = tf.nn.xw_plus_b(hidden, w, b)

        y_prob = tf.nn.softmax(logits)
        y_p = tf.cast(tf.argmax(logits, 1), tf.int32)
        l2_cost = tf.constant(0.0)
        l2_cost += tf.nn.l2_loss(w)
        l2_cost += tf.nn.l2_loss(b)
        softmax_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.in_y))
        cost = softmax_cost

        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        # Accuracy
        check_prediction = tf.equal(y_p, self.in_y)
        acc_cnt = tf.reduce_sum(tf.cast(check_prediction, tf.int32))
        # acc = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
        return y_prob, y_p, cost, train_op, acc_cnt

    def train(self, sess, x, y, seq_len, batch_size, epoch_size):
        learning_rate = 1e-3
        for epoch in range(epoch_size):
            total_cost = 0.0
            total_batch = 0
            total_acc_num = 0
            for batch_x, batch_y, batch_len in batch_iter(x, y, seq_len, batch_size, shuffle=True):
                total_batch += 1
                _, cost_val, acc_cnt = sess.run([self.train_op, self.cost, self.acc_cnt],
                                                feed_dict={self.in_x: batch_x,
                                                           self.in_y: batch_y,
                                                           self.in_len: batch_len,
                                                           self.learning_rate: learning_rate})
                total_acc_num += acc_cnt
                total_cost += cost_val
                if total_batch % 30 == 0:
                    print ('batch_%d cost_val: %0.5f' % (total_batch, cost_val))
            print('Epoch:', '%02d' % (epoch + 1),
                  'cost_avg =', '%0.5f' % (total_cost / total_batch),
                  'acc: %0.5f' % (total_acc_num/(0.0+len(x))))
            if epoch + 1 < 4:
                learning_rate /= (10*epoch+10)
            self.saver.save(sess, config.save_dir + '/rcnn_saver.ckpt', global_step=epoch + 1)

    def test(self, sess, x, y, seq_len):
        ckpt = tf.train.get_checkpoint_state(config.save_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('no rcnn model ...')
        total_acc_num = 0
        for batch_x, batch_y, batch_len in batch_iter(x, y, seq_len, 100):
            acc_cnt = sess.run(self.acc_cnt, feed_dict={self.in_x: batch_x,
                                                        self.in_y: batch_y,
                                                        self.in_len: batch_len})
            total_acc_num += acc_cnt
        print ('test acc: %0.5f' % (total_acc_num/(0.0+len(x))))

    def predict(self, sess, x, y, seq_len):
        ckpt = tf.train.get_checkpoint_state(config.save_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('no rcnn model ...')
        probability = []
        for batch_x, batch_y, batch_len in batch_iter(x, y, seq_len, 100):
            proba = sess.run(self.y_prob, feed_dict={self.in_x: batch_x,
                                                     self.in_y: batch_y,
                                                     self.in_len: batch_len})
            probability.append(proba)
        return np.array(probability)

    def train_and_dev(self, sess, x, y, seq_len, batch_size, test_x, test_y, test_seq_len, epoch_size):
        learning_rate = 1e-3
        for epoch in range(epoch_size):
            total_cost = 0.0
            total_batch = 0
            total_acc_num = 0
            for batch_x, batch_y, batch_len in batch_iter(x, y, seq_len, batch_size, shuffle=True):
                total_batch += 1
                _, cost_val, acc_cnt = sess.run([self.train_op, self.cost, self.acc_cnt],
                                                feed_dict={self.in_x: batch_x,
                                                           self.in_y: batch_y,
                                                           self.in_len: batch_len,
                                                           self.learning_rate: learning_rate})
                total_acc_num += acc_cnt
                total_cost += cost_val
                if total_batch % 30 == 0:
                    print ('batch_%d cost_val: %0.5f' % (total_batch, cost_val))
            print('Epoch:', '%02d' % (epoch + 1),
                  'cost_avg =', '%0.5f' % (total_cost / total_batch),
                  'acc: %0.5f' % (total_acc_num/(0.0+len(x))))
            if epoch + 1 < 4:
                learning_rate /= (10*epoch+10)
            self.saver.save(sess, config.save_dir + '/rcnn_saver.ckpt', global_step=epoch + 1)

            ###
            total_acc_num_test = 0
            for batch_test_x, batch_test_y, batch_len in batch_iter(test_x, test_y, test_seq_len, 100):
                acc_cnt_test = sess.run(self.acc_cnt, feed_dict={self.in_x: batch_test_x,
                                                                 self.in_y: batch_test_y,
                                                                 self.in_len: batch_len})
                total_acc_num_test += acc_cnt_test
            print('test acc: %0.5f' % (total_acc_num_test / (0.0 + len(test_x))))


def main():
    data = Data(load=False)
    model = Model(data.we)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # model.train(sess, x=data.train_x, y=data.train_y, seq_len=data.train_seq_len, batch_size=128, epoch_size=10)
    # model.test(sess, data.dev_x, data.dev_y, data.dev_seq_len)
    model.train_and_dev(sess, data.train_x, data.train_y, data.train_seq_len, 128,
                        data.dev_x, data.dev_y, data.dev_seq_len, epoch_size=10)

    # probability = model.predict(sess, data.test_x, data.test_y, data.test_seq_len)


if __name__ == '__main__':
    main()
