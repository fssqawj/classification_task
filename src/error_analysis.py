# coding: utf-8
import codecs
import csv
import sys, argparse

if __name__ == '__main__':
    file_path = sys.argv[1]
    outf = sys.argv[2]
    with open(file_path) as f, codecs.open(outf, 'w+', 'utf_8_sig') as foutcsv:
        fout = csv.writer(foutcsv)
        fout.writerow(['pred', 'ground', 'sents'])
        for line in f:
            pred, sent = line.split('\t#\t')
            sent = sent.split('\t')
            ground = sent[0]
            sentstr = '\t'.join(sent[1:])
            if pred != ground:
                fout.writerow([pred, ground, sentstr])
