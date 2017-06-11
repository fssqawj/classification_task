# coding: utf8
from __future__ import print_function
import json, pickle
import numpy as np

from stst import config
from stst import utils
from stst.features import Feature
from stst.lib.kernel import vector_kernel as vk


class UnigramFeature(Feature):

    # def __init__(self, **kwargs):
    #     super(UnigramFeature).__init__(**kwargs)

    def extract_information(self, train_instances):
        if self.is_training:
            sents = []
            for train_instance in train_instances:
                sent = train_instance.get_word()
                sents.append(sent)
            idf_dict = utils.idf_calculator(sents)

            #idf_dict = sorted(idf_dict.iteritems(), key=lambda x: x[1], reverse=True)

            with utils.create_write_file(config.DICTIONARY_DIR + '/idf_dict.txt') as fw:
                for key in idf_dict:
                    print('{}\t{}'.format(key, idf_dict[key]), file=fw)

            print(len(idf_dict))
        else:
            with utils.create_read_file(config.DICTIONARY_DIR + '/idf_dict.txt') as fr:
                idf_dict = {}
                for line in fr:
                    line = line.strip().split('\t')
                    idf_dict[line[0]] = float(line[1])

        self.unigram_dict = idf_dict

    def extract(self, train_instance):
        sent = train_instance.get_word()
        feats = utils.vectorize(sent, self.unigram_dict)
        infos = [len(self.unigram_dict), 'unigram']
        return feats, infos


class FuckFeature(Feature):

    # def __init__(self, **kwargs):
    #     super(UnigramFeature).__init__(**kwargs)

    def extract_information(self, train_instances):
        if self.is_training:
            sents, labels = [], []
            for train_instance in train_instances:
                sent = train_instance.get_word()
                label = train_instance.get_label()
                sents.append(sent)
                labels.append(label)

            rf_dict = utils.rf_calculator(sents, labels, max_cnt=1000)
            with utils.create_write_file(config.DICTIONARY_DIR + '/rf_dict.txt', 'w') as fw:
                json.dump(rf_dict, fw, ensure_ascii=False)

        with utils.create_read_file(config.DICTIONARY_DIR + '/rf_dict.txt', 'rb') as fr:
            rf_dict = json.load(fr)

        with utils.create_read_file(config.DICTIONARY_DIR + '/vocab.txt') as fr:
            vocab_dict = {}
            for line in fr:
                line = line.strip().split('\t')
                vocab_dict[line[0]] = int(line[1])

        self.rf_dict = rf_dict
        self.vocab_dict = vocab_dict

    def extract(self, train_instance):
        sent = train_instance.get_word()
        feats = [0] * 18
        for word in sent:
            for label in range(18):
                s_label = str(label)
                if word not in self.rf_dict[s_label]:
                    continue
                if word not in self.vocab_dict:
                    continue

                rf = self.rf_dict[s_label][word]
                cnt = self.vocab_dict[word]
                feats[label] += rf / cnt
        infos = [18, 'fuck']
        return feats, infos


class TFFeature(Feature):

    def __init__(self, type='word', convey='count', **kwargs):
        super(TFFeature, self).__init__(**kwargs)
        self.type = type
        self.convey = convey
        self.feature_name = '{}-{}#{}'.format(self.feature_name, type, self.convey)

    def extract_information(self, train_instances):
        if self.is_training:
            sents = []
            for train_instance in train_instances:
                sent = train_instance.get_sent(self.type)
                sents.append(sent)
            idf_dict = utils.idf_calculator(sents)
            with utils.create_write_file(config.DICTIONARY_DIR + '/{}_idf_dict.txt'.format(self.type)) as fw:
                idf_dict_tuple = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
                for key, value in idf_dict_tuple:
                    print('{}\t{}'.format(key, value), file=fw)
        else:
            with utils.create_read_file(config.DICTIONARY_DIR + '/{}_idf_dict.txt'.format(self.type)) as fr:
                idf_dict = {}
                for line in fr:
                    line = line.strip().split('\t')
                    idf_dict[line[0]] = float(line[1])
        self.unigram_dict = idf_dict
        word_keys = sorted(idf_dict.keys(), reverse=True)
        self.word2index = {word: i for i, word in enumerate(word_keys)}

    def extract(self, train_instance):
        sent = train_instance.get_sent(self.type)
        feats = utils.sparse_vectorize(sent, self.unigram_dict, self.word2index, self.convey)
        infos = [len(self.unigram_dict), 'tf']
        feats = Feature._feat_dict_to_string(feats)
        return feats, infos


class BigramFeature(Feature):

    def __init__(self, type='word', convey='count', **kwargs):
        super(BigramFeature, self).__init__(**kwargs)
        self.type = type
        self.convey = convey
        self.feature_name = '{}-{}#{}'.format(self.feature_name, type, self.convey)

    def extract_information(self, train_instances):
        if self.is_training:
            sents = []
            for train_instance in train_instances:
                sent = train_instance.get_sent(self.type)
                sent = utils.make_ngram(sent, 2)
                sents.append(sent)
            idf_dict = utils.idf_calculator(sents)
            with utils.create_write_file(config.DICTIONARY_DIR + '/{}_bigram_dict.txt'.format(self.type)) as fw:
                idf_dict_tuple = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
                for key, value in idf_dict_tuple:
                    print('{}\t{}\t{}'.format(key[0], key[1], value), file=fw)
        else:
            with utils.create_read_file(config.DICTIONARY_DIR + '/{}_bigram_dict.txt'.format(self.type)) as fr:
                idf_dict = {}
                for line in fr:
                    line = line.strip().split('\t')
                    idf_dict[(line[0], line[1])] = float(line[2])
        self.bigram_dict = idf_dict
        word_keys = sorted(idf_dict.keys(), reverse=True)
        self.word2index = {word: i for i, word in enumerate(word_keys)}

    def extract(self, train_instance):
        sent = train_instance.get_sent(self.type)
        sent = utils.make_ngram(sent, 2)
        feats = utils.sparse_vectorize(sent, self.bigram_dict, self.word2index, self.convey)
        infos = [len(self.bigram_dict), 'bigram']
        feats = Feature._feat_dict_to_string(feats)
        return feats, infos


def pooling(word_embs, dim, pooling_types):
    if pooling_types == 'avg':
        function = np.average
    elif pooling_types == 'min':
        function = np.amin
    elif pooling_types == 'max':
        function = np.amax
    else:
        print(pooling_types)
        raise NotImplementedError

    if len(word_embs) == 0:
        vec = np.zeros((dim,))
    else:
        vec = function(word_embs, axis=0)
    return vec


def minavgmaxpooling(word_list, word2index, embeddings, dim, convey, idf_weight, pooling_type='avg'):
    word_embs = []
    for word in word_list:
        # gain word vector
        w2v = embeddings[word2index[word]]

        # gain weight of word
        # !!! be careful, very slow
        # default_idf_weight = min(idf_weight.values())
        if convey == 'idf':
            w = idf_weight.get(word, 0.0) # default_idf_weight
        elif convey == 'count':
            w = 1.0
        else:
            raise NotImplementedError

        # append
        w2v = w * np.array(w2v)
        word_embs.append(w2v)

    # concat sentence embedding
    vecs = []
    if pooling_type == 'all':
        pooling_types = ['avg', 'min', 'max']
    elif pooling_type == 'avg':
        pooling_types = ['avg']
    else:
        raise NotImplementedError

    for pooling_type in pooling_types:   # ['avg', 'min', 'max']:
        vec = pooling(word_embs, dim, pooling_type)
        vecs.append(vec)
    vecs = np.reshape(vecs, [-1])
    return vecs


class MinAvgMaxEmbeddingFeature(Feature):
    def __init__(self, emb_name, dim, emb_file, pooling_type='avg', binary=False, **kwargs):
        super(MinAvgMaxEmbeddingFeature, self).__init__(**kwargs)
        self.emb_name = emb_name
        self.dim = dim
        self.emb_file = emb_file
        self.binary = binary
        self.type = 'word'
        self.pooling_type = pooling_type
        self.feature_name = self.feature_name + '-%s-%s' % (emb_name, pooling_type)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            sent = train_instance.get_sent(self.type)
            seqs.append(sent)

        self.idf_weight = utils.idf_calculator(seqs)
        self.word2index = {word:index for index, word in enumerate(self.idf_weight.keys())}
        self.embeddings = utils.load_word_embedding(self.word2index, self.emb_file, self.dim, self.binary)


    def extract(self, train_instance):
        sent = train_instance.get_sent(self.type)
        vec_sent = minavgmaxpooling(sent, self.word2index, self.embeddings, self.dim,
                                          convey='count', idf_weight=self.idf_weight,
                                          pooling_type=self.pooling_type)
        features = vec_sent.tolist()
        infos = [self.emb_name, self.type, self.pooling_type]
        return features, infos


