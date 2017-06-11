import sys
import argparse
import csv
import codecs


def split(line):
    line = line.strip()
    line = line.split('\t#\t')
    pred = line[0]

    sent = line[1].split('\t')
    ground = sent[0]
    sent = '\t'.join(sent[1:])
    return pred, ground, sent


def diff(file_1, file_2, output_file):
    f1 = open(file_1).readlines()
    f2 = open(file_2).readlines()

    with codecs.open(output_file, 'w', 'utf_8_sig') as csv_file:

        writter = csv.writer(csv_file, delimiter=',')

        for line_1, line_2 in zip(f1, f2):
            pred_1, ground_1, sent_1 = split(line_1)
            pred_2, ground_2, sent_2 = split(line_2)

            assert ground_1 == ground_2 and sent_1 == sent_2
            if pred_1 == ground_1 and pred_2 != ground_2:
                writter.writerow([pred_1, pred_2, ground_1, sent_1])


if __name__ == '__main__':
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]
    output_file = sys.argv[3]

    diff(file_1, file_2, output_file)