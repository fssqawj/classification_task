import sys
from stst.evaluation import Evaluation

if __name__ == '__main__':
    gold_file = sys.argv[1]
    predict_file = sys.argv[2]
    Evaluation(gold_file, predict_file)