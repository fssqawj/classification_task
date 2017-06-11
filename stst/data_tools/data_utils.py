# coding: utf8
from __future__ import print_function

import codecs
import json
import os

import pyprind
import jieba
from stst import utils
from stst.data_tools.sent import Sent
from stst.config import DICT_LABEL_TO_INDEX

def load_data(train_file):
    """
    Return list of dataset given train_file
    Value: [(sent:str, label:int), ... ]
    """
    with codecs.open(train_file, 'r', encoding='utf8') as f:
        data = []
        for idx, line in enumerate(f):
            line = line.strip().split('\t')
            label = DICT_LABEL_TO_INDEX[line[0]]
            # sent = line[1].split()
            sent = line[1]
            data.append((sent, label))
    return data

def preprocess(sent):
    if type(sent) is list:
        raise NotImplementedError
    sent = list(jieba.cut(sent))
    sent = ' '.join(sent).split()
    return sent

def load_parse_data(train_file, nlp=None, flag=False):
    """
    Load data after Parse, like POS, NER, etc.
    Value: [ SentPair:class, ... ]
    Parameter:
        flag: False(Default), Load from file (resources....)
              True, Parse and Write to file, and then load from file
    """
    ''' Pre-Define Write File '''

    # parse_train_file = config.PARSE_DIR + '/' + \
    #                    utils.FileManager.get_file(train_file)

    parse_train_file = train_file.replace('./data', './generate/parse')

    if flag or not os.path.isfile(parse_train_file):

        print(train_file)

        ''' Parse Data '''
        data = load_data(train_file)

        print('*' * 50)
        print("Parse Data, train_file=%s, n_train=%d\n" % (train_file, len(data)))

        parse_data = []
        process_bar = pyprind.ProgPercent(len(data))
        for (sent, label) in data:
            process_bar.update()
            sent = preprocess(sent)
            parse_data.append((sent, label))

        ''' Write Data to File '''
        with utils.create_write_file(parse_train_file) as f_parse:
            for parse_instance in parse_data:
                line = json.dumps(parse_instance, ensure_ascii=False)
                print(line, file=f_parse)

    ''' Load Data from File '''
    print('*' * 50)
    parse_data = []
    with utils.create_read_file(parse_train_file) as f:
        for line in f:
            sent, label = json.loads(line)
            sentpair_instance = Sent(sent, label)
            parse_data.append(sentpair_instance)

    print("Load Data, train_file=%s, n_train=%d\n" % (train_file, len(parse_data)))
    return parse_data