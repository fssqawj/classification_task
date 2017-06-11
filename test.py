# coding: utf-8
from __future__ import print_function
from stst.config import DICT_LABEL_TO_INDEX, DICT_INDEX_TO_LABEL
import json

def dev():
    inp = './generate/outputs/S2-gb/dev-exp-fix.txt'
    inp_real = './data/nlpcc_data/word/dev-exp-fix.txt'
    oup = './generate/outputs/S2-gb/dev-exp-fix-prob.txt'

    fr = open(inp, 'r')
    fr_real = open(inp_real, 'r')
    fw = open(oup, 'w')

    headline = fr.readline()
    headline = headline.strip().split()

    index2pos = {int(headline[i+1]):i + 1 for i in range(18)}
    pos2index = {i + 1:int(headline[i+1]) for i in range(18)}
    # print(fw, fr.readline())


    pred_labels = []
    pred_probs = []
    real_labels = []
    real_probs = []
    real_sents = []

    for line in fr_real:
        line = line.strip().split()
        real_labels.append(line[0])
        real_sents.append(' '.join(line[1:]))

    cnt = 0
    probs = []
    for line in fr:
        line = line.strip().split()
        # print(line)
        prob = {}
        prob['gold'] = real_labels[cnt]
        prob['predict'] = DICT_INDEX_TO_LABEL[int(line[0])]
        for i in range(0, 18):
            label = DICT_INDEX_TO_LABEL[pos2index[i+1]]
            label_prob = float(line[i+1])
            prob[label] = label_prob
        pred_labels.append(DICT_INDEX_TO_LABEL[int(line[0])])
        pred_probs.append(line[index2pos[int(line[0])]])
        real_probs.append(line[index2pos[DICT_LABEL_TO_INDEX[real_labels[cnt]]]])
        # print(prob)
        # exit(1)
        cnt = cnt + 1
        probs.append(prob)

    for a, s, d, f, g in zip(pred_labels, pred_probs, real_labels, real_probs, real_sents):
        if a == d:
            fw.write('\n')
        else:
            fw.write('{}\t{}\t{}\t{}\t{}\n'.format(a, str(s), d, str(f), g))

    output_json = './generate/outputs/S2-gb/dev-exp-fix-prob.json'
    fw = open(output_json, 'w')
    for prob in probs:
        print(json.dumps(prob), file=fw)


def test():
    inp = './generate/outputs/S2-gb/test-exp-fix.txt.prob'
    inp_real = './data/nlpcc_data/word/test-exp-fix.txt'
    oup = './generate/outputs/S2-gb/test-exp-fix-prob.txt'

    fr = open(inp, 'r')
    fr_real = open(inp_real, 'r')
    fw = open(oup, 'w')

    headline = fr.readline()
    headline = headline.strip().split()

    index2pos = {int(headline[i + 1]): i + 1 for i in range(18)}
    pos2index = {i + 1: int(headline[i + 1]) for i in range(18)}
    # print(fw, fr.readline())

    pred_labels = []
    pred_probs = []
    real_labels = []
    real_probs = []
    real_sents = []

    for line in fr_real:
        line = line.strip().split()
        real_labels.append(line[0])
        real_sents.append(' '.join(line[1:]))

    cnt = 0
    probs = []
    for line in fr:
        line = line.strip().split()
        # print(line)
        prob = {}
        prob['gold'] = real_labels[cnt]
        prob['predict'] = DICT_INDEX_TO_LABEL[int(line[0])]
        for i in range(0, 18):
            label = DICT_INDEX_TO_LABEL[pos2index[i + 1]]
            label_prob = float(line[i + 1])
            prob[label] = label_prob
        pred_labels.append(DICT_INDEX_TO_LABEL[int(line[0])])
        pred_probs.append(line[index2pos[int(line[0])]])
        real_probs.append(line[index2pos[DICT_LABEL_TO_INDEX[real_labels[cnt]]]])
        # print(prob)
        # exit(1)
        cnt = cnt + 1
        probs.append(prob)

    for a, s, d, f, g in zip(pred_labels, pred_probs, real_labels, real_probs, real_sents):
        if a == d:
            fw.write('\n')
        else:
            fw.write('{}\t{}\t{}\t{}\t{}\n'.format(a, str(s), d, str(f), g))

    output_json = './generate/outputs/S2-gb/test-exp-fix-prob.json'
    fw = open(output_json, 'w')
    for prob in probs:
        print(json.dumps(prob), file=fw)

# test()
if __name__ == '__main__':

    import jieba
    def preprocess(sent):
        if type(sent) is list:
            raise NotImplementedError
        sent = list(jieba.cut(sent))
        sent = ' '.join(sent).split()
        return sent

    dev_label_file = './data/nlpcc_data/word/test.txt'
    dev_file = './data/nlpcc_data/word/test-exp.txt'
    idx = 0
    with open(dev_file, 'r') as fin, open(dev_file.replace('exp', 'exp-fix-2'), 'w') as fout:
        dev_labels = open(dev_label_file).readlines()
        cnt = 0
        for line in fin:
            line = line.strip().split('####')
            sent = line[-1]

            cls = dev_labels[idx].split('\t')[0]
            idx = idx + 1

            # fout.write(cls.replace('#', '') + '\t')
            sents = [sent]
            for x in line[:-1]:
                sents.append(x)
            if len(line) < 2:
                # print(line[0].decode('utf8'))
                sents.append(sent)
                cnt += 1

            w_sents = []
            for sent in sents:
                sent = preprocess(sent)
                sent = '\t'.join(sent)
                w_sents.append(sent)
            fout.write(cls + '\t' + '\t'.join(w_sents) + '\n')
        print(cnt)
