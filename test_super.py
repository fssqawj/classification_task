# coding: utf8
from __future__ import print_function
import stst
import numpy as np
import pickle, json

from stst.config import DICT_LABEL_TO_INDEX, DICT_INDEX_TO_LABEL
from main_tools import dev_file, test_file

def dev():
    dl_prob_file = './generate/outputs/DL/probability.pkl'
    dl_probs = pickle.load(open(dl_prob_file, 'rb'), encoding='latin1')

    xgb_prob_file = './generate/outputs/S2-xgb/dev-exp-fix.txt.pkl'
    xgb_probs = pickle.load(open(xgb_prob_file, 'rb'), encoding='latin1')

    nlp_prob_file = './generate/outputs/S2-gb/dev-exp-fix-prob.json'
    nlp_probs = []
    for line in open(nlp_prob_file):
        json_object = json.loads(line.strip())
        nlp_prob = np.zeros((18,))
        for i in range(18):
            label = DICT_INDEX_TO_LABEL[i]
            prob = float(json_object[label])
            nlp_prob[i, ] = prob
        nlp_probs.append(nlp_prob)
    nlp_probs = np.array(nlp_probs)

    final_probs = dl_probs + nlp_probs + xgb_probs + dl_probs
    final_results = np.argmax(final_probs, axis=1)

    final_file = './generate/outputs/DL/dev-final.txt'

    with open(final_file, 'w') as fw:
        for index in final_results:
            label = DICT_INDEX_TO_LABEL[index]
            print(label, file=fw)

    acc, _, _, _ = stst.Evaluation(dev_file, final_file)
    print(acc)


def test():
    dl_prob_file = './generate/outputs/DL/test_probability.pkl'
    dl_probs = pickle.load(open(dl_prob_file, 'rb'), encoding='latin1')

    xgb_prob_file = './generate/outputs/S2-xgb/test-exp-fix.txt.pkl'
    xgb_probs = pickle.load(open(xgb_prob_file, 'rb'), encoding='latin1')

    nlp_prob_file = './generate/outputs/S2-gb/test-exp-fix-prob.json'
    nlp_probs = []

    for line in open(nlp_prob_file):
        json_object = json.loads(line.strip())
        nlp_prob = np.zeros((18,))
        for i in range(18):
            label = DICT_INDEX_TO_LABEL[i]
            prob = float(json_object[label])
            nlp_prob[i, ] = prob
        nlp_probs.append(nlp_prob)
    nlp_probs = np.array(nlp_probs)


    final_probs = dl_probs + nlp_probs + xgb_probs + dl_probs
    final_results = np.argmax(final_probs, axis=1)

    final_file = './generate/outputs/DL/test-final.txt'

    with open(final_file, 'w') as fw:
        for index in final_results:
            label = DICT_INDEX_TO_LABEL[index]
            print(label, file=fw)

    acc, _, _, _ = stst.Evaluation(test_file, final_file)
    print(acc)

    inp_real = './data/nlpcc_data/word/test-exp-fix.txt'
    fr_real = open(inp_real, 'r')

    real_sents = []
    for line in fr_real:
        line = line.strip().split()
        real_sents.append(' '.join(line[1:]))
    final_file = './generate/outputs/DL/test-check.txt'
    with open(final_file, 'w') as fw:
        for index, sent in zip(final_results, real_sents):
            label = DICT_INDEX_TO_LABEL[index]
            print('{}\t{}'.format(label,sent), file=fw)

test()