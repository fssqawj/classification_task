# coding: utf-8
from __future__ import print_function

import os
import pickle

import numpy as np

np.random.seed(1234)
from config import category2id
import config

def load_stopwords(in_file):
    stop_words = set([])
    with open(in_file, 'r') as fr:
        for line in fr:
            stop_words.add(line.strip())
    return stop_words

g_stop_words = load_stopwords(config.stopwords_file)


def save_params(params, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as fw:
        pickle.dump(params, fw, protocol=pickle.HIGHEST_PROTOCOL)


def load_params(fname):
    if not os.path.exists(fname):
        raise RuntimeError('no file: %s' % fname)
    with open(fname, 'rb') as fr:
        params = pickle.load(fr)
    return params


def load_embed_from_text(in_file, token_dim):
    """
    :return: embed numpy, vocab2id dict
    """
    embed = []
    vocab2id = {}
    print('==> loading embed from txt')
    word_id = 0
    embed.append([0.0] * token_dim)
    with open(in_file, 'r') as fr:
        print('embedding info: ', fr.readline())
        for line in fr:
            t = line.rstrip().split()
            word_id += 1
            # if word_id > 62593:
            #     break
            vocab2id[t[0]] = word_id
            embed.append(list(map(float, t[1:]))) # python3 map return a generator not a list
    print('==> finished load input embed from txt')
    return np.array(embed, dtype=np.float32), vocab2id


def sentence2id_and_pad(in_file, vocab2id, max_len):
    x = []
    y = []
    seq_len = []
    with open(in_file, 'r') as fr:
        for line in fr:
            t = line.split('\t')
            y.append(category2id[t[0]])
            words = [w for w in t[1].split() if w not in g_stop_words]
            words = words[: max_len]
            seq_len.append(len(words))
            x.append([vocab2id.get(w, 0) for w in words] + [0]*(max_len-len(words)))
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32), np.array(seq_len, dtype=np.float32)


def batch_iter(x, y, seq_len, batch_size, shuffle=False):
    assert len(x) == len(y)
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(len(x))
    for start_idx in range(0, len(x), batch_size):
        excerpt = idx[start_idx:start_idx + batch_size]
        yield x[excerpt], y[excerpt], seq_len[excerpt]


class Data(object):
    def __init__(self, load=False):
        self.load = load
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.dev_file = config.dev_file
        self.word_embed_file = config.word_embed_file
        self.stop_words_file = config.stopwords_file
        self.vocab2id_file = config.vocab2id_file
        self.we_file = config.we_file
        self.word_dim = config.word_dim
        self.max_len = config.max_sent_len

        self.we = None
        self.train_x = None
        self.train_y = None
        self.train_seq_len = None
        self.test_x = None
        self.test_y = None
        self.test_seq_len = None
        self.dev_x = None
        self.dev_y = None
        self.dev_seq_len = None
        self.init_data()

    def init_data(self):
        if self.load:
            self.we = load_params(self.we_file)
            vocab2id = load_params(self.vocab2id_file)
        else:
            self.we, vocab2id = load_embed_from_text(self.word_embed_file, self.word_dim)
            save_params(self.we, self.we_file)
            save_params(vocab2id, self.vocab2id_file)

        self.train_x, self.train_y, self.train_seq_len = sentence2id_and_pad(self.train_file, vocab2id, self.max_len)
        self.dev_x, self.dev_y, self.dev_seq_len = sentence2id_and_pad(self.dev_file, vocab2id, self.max_len)
        # self.test_x, self.test_y, self.test_seq_len = sentence2id_and_pad(self.test_file, vocab2id, self.max_len)
        print ("vocab size: %d" % len(vocab2id), "we shape: ", self.we.shape)
        print ("train_x: %d " % len(self.train_x), "train_y: %d" % len(self.train_y))
        # print ("test_x: %d " % len(self.test_x), "test_y: %d" % len(self.test_y))
        print ("dev_x: %d " % len(self.dev_x), "dev_y: %d" % len(self.dev_y))


def test():
    # data = Data()
    # print(data.dev_x[-3:])
    # print(data.dev_seq_len[-3:])
    stop_words = load_stopwords('../data/nlpcc_data/stopwords.txt')
    print (len(stop_words))
    for s in stop_words:
        print (s)
    my_in_file = '../headlines-dev.txt'
    with open(my_in_file, 'r') as fr:
        for line in fr:
            t = line.split('\t')
            print (category2id[t[0]])
            print (t[1])
            for w in t[1].split():
                if w in stop_words:
                    print(w)
            raw_input()


if __name__ == '__main__':
    test()
