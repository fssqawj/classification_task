# coding: utf-8
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import config
from data import Data, batch_iter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # for issue: The TensorFlow library wasn't compiled to use SSE3


class Model(object):
    def __init__(self, we, params=None):
        self.seq_len = config.max_sent_len
        self.embed_size = 100
        self.class_num = 18
        self.lstm_size = 100
        self.grad_clip = 30
        self.learning_rate = tf.placeholder(tf.float32)
        tf.set_random_seed(1234)

        self.in_x = tf.placeholder(tf.int32, [None, None])  # shape: (batch x seq)
        self.in_y = tf.placeholder(tf.int32, [None])
        self.in_len = tf.placeholder(tf.int32, [None])

        self.in_char = tf.placeholder(tf.int32, [None, None, None])
        self.in_char_len = tf.placeholder(tf.int32, [None, None])

        self.aux_xs = tf.placeholder(tf.int32, [None, None, None])
        self.aux_xs_len = tf.placeholder(tf.int32, [None, None])
        self.aux_len = tf.placeholder(tf.int32, [None])

        self.dropout_rate = tf.placeholder(tf.float32, None)

        self.we = tf.Variable(we, name='emb')

        self.y_prob, self.y_p, self.cost, self.train_op, self.acc_cnt = self._build_model(params)
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self, params=None):

        rcnn_size = self.lstm_size*2 + self.embed_size

        origin_sent = self.build_left()  # [batch_size, rcnn_size]
        expand_sents = self.build_right()  # [batch_size, sent_number, rcnn_size]

        expand_mask = tf.sequence_mask(self.aux_len, dtype=tf.float32)

        expand_sents = self.local_gate_mechanism(origin_sent, expand_sents, expand_mask)

        expand_sent = self.attention_mechanism(origin_sent, expand_sents, expand_mask)
        # expand_sent = self.self_attention_mechanism(expand_sents, expand_mask)

        # pooled = tf.concat([origin_sent, expand_sent], axis=1)
        # rcnn_size *= 2

        pooled = self.gate_mechanism(origin_sent, expand_sent)

        hidden_size = 50
        W_h = tf.Variable(tf.random_normal([rcnn_size, hidden_size], stddev=0.01))
        b_h = tf.Variable(tf.zeros([hidden_size]))
        pooled = tf.nn.xw_plus_b(pooled, W_h, b_h)
        pooled = tf.nn.tanh(pooled)

        w = tf.Variable(tf.random_normal([hidden_size, self.class_num], stddev=0.01))
        b = tf.Variable(tf.zeros([self.class_num]))
        logits = tf.nn.xw_plus_b(pooled, w, b)
        y_prob = tf.nn.softmax(logits)
        y_p = tf.cast(tf.argmax(logits, 1), tf.int32)

        softmax_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.in_y))
        cost = softmax_cost

        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1 and v.name != 'emb'])
        # cost += 1e-5 * l2_loss

        # train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, tvars))

        # Accuracy
        check_prediction = tf.equal(y_p, self.in_y)
        acc_cnt = tf.reduce_sum(tf.cast(check_prediction, tf.int32))
        # acc = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
        return y_prob, y_p, cost, train_op, acc_cnt


    def build_left(self):
        '''

        :return: Tensor, [batch_size, hidden_size]
        '''
        # shape: (batch x time_step x word_dim)
        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_x)

        # Create a lstm layer
        with tf.variable_scope('BiLSTM_Layer', reuse=None):
            # stack lstm : tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)
            # Get layer activations (second output is the final state of the layer, do not need)
            # [batch, time_step, n_hidden]
            cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_size)

            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=(1 - self.dropout_rate))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=(1 - self.dropout_rate))

            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedded_seq,
                                                                  self.in_len, dtype=tf.float32)

            context = tf.concat([b_outputs[0], embedded_seq, b_outputs[1]], axis=2)
            context = tf.expand_dims(context, -1)
            pooled = tf.nn.max_pool(context,
                                    ksize=[1, self.seq_len, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="pool")
            pooled = tf.squeeze(pooled)
        return pooled

    def build_right(self):
        '''
        :return: Tensor, [batch_size, aux_sent_number, hidden_size]
        '''

        # self.aux_xs_len = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
        # self.aux_xs = tf.placeholder(tf.int32, [None, None, None])  # [batch_size, question_len, q_char_len]

        input_shape = tf.shape(self.aux_xs)
        print(input_shape)
        batch_size = input_shape[0]
        sent_number = input_shape[1]
        word_number = input_shape[2]

        # [batch_size, question_len, q_char_len, char_dim]
        word_repres = tf.nn.embedding_lookup(self.we, self.aux_xs)

        # [batch_size*question_len, q_char_len, char_dim]
        word_repres = tf.reshape(word_repres, shape=[-1, word_number, self.embed_size])
        word_lengths = tf.reshape(self.aux_xs_len, [-1])

        with tf.variable_scope('BiLSTM_Layer2', reuse=None):
            cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_size)

            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=(1 - self.dropout_rate))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=(1 - self.dropout_rate))

            # question_representation
            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, word_repres,
                                                       sequence_length=word_lengths, dtype=tf.float32)
            # [batch_size*question_len, q_char_len, char_lstm_dim]

            context = tf.concat([b_outputs[0], word_repres, b_outputs[1]], axis=2)

            # context = tf.reshape(context, [batch_size, sent_number, word_number, self.embed_size + 2 * self.lstm_size])

            context = tf.expand_dims(context, -1)

            pooled = tf.nn.max_pool(context,
                                    ksize=[1, self.seq_len, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="pool")

            pooled = tf.squeeze(pooled)
            outputs = pooled
            outputs = tf.reshape(outputs, [batch_size, sent_number, self.embed_size + 2 * self.lstm_size])
        return outputs


    def simple_sum(self, passage_repres, passage_mask):
        # passage_output = tf.reduce_sum(passage_repres, axis=1)
        passage_output = tf.reduce_sum(passage_repres, axis=1) / tf.reduce_sum(passage_mask, axis=1, keep_dims=True)
        return passage_output

    def attention_mechanism(self, question_rep, passage_repres, passage_mask):
        """

        :param question_rep: [batch_size, hidden_size]
        :param passage_repres: [batch_size, sent_number, hidden_size]
        :param passage_mask: [batch_size, sent_number]
        :return:
        """
        out_size = self.embed_size + self.lstm_size * 2

        with tf.variable_scope("attention", reuse=None):
            # W_bilinear = tf.get_variable("W_bilinear", [out_size, out_size], dtype=tf.float32)
            # b_bilinear = tf.get_variable("b_bilinear", [out_size, ], dtype=tf.float32)
            # question_rep = tf.matmul(question_rep, W_bilinear) + b_bilinear
            question_rep = tf.expand_dims(question_rep, 1)

            # passage_repres: [batch_size, sent_number, hidden_size]
            # question_rep: [batch_size, 1, hidden_size]
            passage_prob = tf.nn.softmax(tf.reduce_sum(passage_repres * question_rep, 2))

            passage_prob = passage_prob * passage_mask
            passage_prob = passage_prob / tf.reduce_sum(passage_prob, -1, keep_dims=True)

            self.attention_prob = passage_prob

            passage_output = tf.reduce_sum(passage_repres * tf.expand_dims(passage_prob, -1), axis=1)

        return passage_output

    def self_attention_mechanism(self, passage_repres, passage_mask):

        out_size = self.embed_size + self.lstm_size * 2

        # v_attention = tf.get_variable("v_attention", [1, out_size], dtype=tf.float32)
        v_attention = tf.reduce_max(passage_repres, 1)


        with tf.variable_scope("self_attention", reuse=None):

            W_bilinear = tf.get_variable("W_bilinear", [out_size, out_size], dtype=tf.float32)
            b_bilinear = tf.get_variable("b_bilinear", [out_size, ], dtype=tf.float32)

            question_rep = tf.matmul(v_attention, W_bilinear) + b_bilinear
            question_rep = tf.expand_dims(question_rep, 1)

            passage_prob = tf.nn.softmax(tf.reduce_sum(passage_repres * question_rep, 2))
            passage_prob = passage_prob * passage_mask / tf.reduce_sum(passage_mask, -1, keep_dims=True)

            self.self_attention_prob = passage_prob
            passage_output = tf.reduce_sum(passage_repres * tf.expand_dims(passage_prob, -1), axis=1)
            # passage_output = tf.reduce_sum(passage_repres, axis=1)
        return passage_output

    def gate_mechanism(self, question_rep, passage_rep):

        output_size = self.embed_size + self.lstm_size * 2

        with tf.variable_scope("gate_layer"):
            gate_question_w = tf.get_variable("gate_question_w", [output_size, output_size], dtype=tf.float32)
            gate_passage_w = tf.get_variable("gate_passage_w", [output_size, output_size], dtype=tf.float32)

            gate_b = tf.get_variable("gate_passage_b", [output_size], dtype=tf.float32)

            gate = tf.nn.sigmoid(tf.matmul(question_rep, gate_question_w) +
                                 tf.matmul(passage_rep, gate_passage_w) + gate_b)

            outputs = question_rep * gate + passage_rep * (1.0 - gate)

        return outputs

    def local_gate_mechanism(self, question_rep, passage_repres, passage_mask):

        output_size = self.embed_size + self.lstm_size * 2

        input_shape = tf.shape(passage_repres)
        print(input_shape)
        batch_size = input_shape[0]
        sent_number = input_shape[1]
        hidden_size = input_shape[2]

        with tf.variable_scope("local_gate_layer"):
            gate_question_w = tf.get_variable("gate_question_w", [output_size, output_size], dtype=tf.float32)

            gate_question = tf.matmul(question_rep, gate_question_w)
            gate_question = tf.expand_dims(gate_question, 1)  # [batch_size, 1, hidden_size]

            gate_passage_w = tf.get_variable("gate_passage_w", [output_size, output_size], dtype=tf.float32)

            gate_b = tf.get_variable("gate_b", [output_size], dtype=tf.float32)
            gate_b = tf.expand_dims(gate_b, 0)
            gate_b = tf.expand_dims(gate_b, 0)

            '''
            passage_repres: [batch_size, sents_number, hidden_size]
            gate_passage_w: [batchc_size, output_size, output_size]
            '''
            passage_rep = tf.reshape(passage_repres, [-1, output_size])
            gate_passage = tf.matmul(passage_rep, gate_passage_w)
            gate_passage = tf.reshape(gate_passage, [batch_size, sent_number, output_size])

            gate = tf.nn.sigmoid(gate_question + gate_passage + gate_b)

            outputs = passage_repres * gate # + tf.expand_dims(question_rep, 1) * (1.0 - gate)

        return outputs

    def word_represent(self):
        '''

        :return: Tensor, [batch_size, sent_Length, embedding_size]
        '''

        pass

    def gate_word_char_represent(self):
        '''

        :return:
        '''
        pass

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

    def train_and_dev(self, sess, x, y, seq_len, batch_size, test_x, test_y, test_seq_len, epoch_size, data=None):
        learning_rate = 1e-3
        train_aux_xs = data.train_aux_xs
        train_aux_xs_len = data.train_aux_xs_len
        train_aux_len = data.train_aux_len

        test_aux_xs = data.dev_aux_xs
        test_aux_xs_len = data.dev_aux_xs_len
        test_aux_len = data.dev_aux_len

        for epoch in range(epoch_size):
            total_cost = 0.0
            total_batch = 0
            total_acc_num = 0
            for batch_x, batch_y, batch_len, batch_xs, batch_xs_len, batch_aux_len in batch_iter(x, y, seq_len, batch_size, train_aux_xs, train_aux_xs_len, train_aux_len, shuffle=True):
                total_batch += 1
                _, cost_val, acc_cnt = sess.run([self.train_op, self.cost, self.acc_cnt],
                                                feed_dict={self.in_x: batch_x,
                                                           self.in_y: batch_y,
                                                           self.in_len: batch_len,
                                                           self.aux_xs:batch_xs,
                                                           self.aux_xs_len:batch_xs_len,
                                                           self.aux_len:batch_aux_len,
                                                           self.learning_rate: learning_rate,
                                                           self.dropout_rate:0.0})
                total_acc_num += acc_cnt
                total_cost += cost_val
                if total_batch % 30 == 0:
                    print('batch_%d cost_val: %0.5f' % (total_batch, cost_val))
            print('Epoch:', '%02d' % (epoch + 1),
                  'cost_avg =', '%0.5f' % (total_cost / total_batch),
                  'acc: %0.5f' % (total_acc_num/(0.0+len(x))))

            if epoch < 4 and epoch % 2 == 1:
                learning_rate /= 10.
                print('drop learning rate, Epoch:{} - {}'.format(epoch + 1, learning_rate))


            # self.saver.save(sess, config.save_dir+'/rcnn_saver.ckpt', global_step=epoch+1)

            ###
            total_acc_num_test = 0
            for batch_test_x, batch_test_y, batch_len, batch_xs, batch_xs_len, batch_aux_len in batch_iter(test_x, test_y, test_seq_len, 200, test_aux_xs, test_aux_xs_len, test_aux_len):
                acc_cnt_test = sess.run(self.acc_cnt, feed_dict={self.in_x: batch_test_x,
                                                                 self.in_y: batch_test_y,
                                                                 self.in_len: batch_len,
                                                                 self.aux_xs: batch_xs,
                                                                 self.aux_xs_len: batch_xs_len,
                                                                 self.aux_len:batch_aux_len,
                                                                 self.dropout_rate:0.0
                                                                 })
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
                        data.dev_x, data.dev_y, data.dev_seq_len, epoch_size=10, data=data)

    # probability = model.predict(sess, data.test_x, data.test_y, data.test_seq_len)


if __name__ == '__main__':
    main()
