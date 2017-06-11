# coding: utf-8
import codecs
import csv
import sys, argparse

if __name__ == '__main__':
    gold_file = sys.argv[1]
    predict_file = sys.argv[2]
    output_file = sys.argv[3]
    with open(gold_file) as f1, open(predict_file) as f2, codecs.open(output_file, 'w+', 'utf_8_sig') as foutcsv:
        fout = csv.writer(foutcsv)
        fout.writerow(['pred', 'ground', 'sents'])
        for line_1, line_2 in zip(f1, f2):
            ground, sent = line_1.split('\t')
            pred, sent = line_2.split('\t')
            if pred != ground:
                fout.writerow([pred, ground, sent])
