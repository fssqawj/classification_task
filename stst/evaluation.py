import scipy.stats as meas


def evaluation(predict, gold):
    """
    pearsonr of predict and gold
    :param predict: list
    :param gold: list
    :return: mape
    """
    pearsonr = meas.pearsonr(predict, gold)[0]
    return pearsonr


def eval_file(predict_file, gold_feature_file):
    predict = open(predict_file).readlines()
    gold = open(gold_feature_file).readlines()
    predict = [float(x.strip().split()[0])for x in predict]
    gold = [float(x.strip().split()[0]) for x in gold]
    pearsonr = evaluation(predict, gold)
    return pearsonr


def eval_output_file(predict_file):
    predict, gold = [], []
    with open(predict_file) as f:
        for line in f:
            line = line.strip().split('\t#\t')
            predict.append(float(line[0]))
            gold.append(float(line[1].split('\t')[0]))
    pearsonr = evaluation(predict, gold)
    return pearsonr


def eval_file_corpus(predict_file_list, gold_file_list):
    predicts, golds = [], []
    for predict_file, gold_file in zip(predict_file_list, gold_file_list):
        predict = open(predict_file).readlines()
        gold = open(gold_file).readlines()
        predicts += predict
        golds += gold
    predicts = [float(x.strip().split()[0]) for x in predicts]
    golds = [float(x.strip().split()[0]) for x in golds]
    pearsonr = evaluation(predicts, golds)
    return pearsonr



######classification###############################################################
# coding: utf-8

from stst.confusion_matrix import Alphabet, ConfusionMatrix
from stst.config import DICT_INDEX_TO_LABEL

def Evaluation(gold_file_path, predict_file_path):
    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:

        gold_list = [ line.strip().split('\t')[0] for line in gold_file]
        predicted_list = [line.strip().split("\t#\t")[0] for line in predict_file]


        binary_alphabet = Alphabet()
        for i in range(18):
            binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

        cm = ConfusionMatrix(binary_alphabet)
        cm.add_list(predicted_list, gold_list)

        cm.print_out()
        macro_p, macro_r, macro_f1 = cm.get_average_prf()
        overall_accuracy = cm.get_accuracy()
        return overall_accuracy, macro_p, macro_r, macro_f1


if __name__ == '__main__':
    pass
