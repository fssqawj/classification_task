from __future__ import print_function
import stst
from features.features_unigram import UnigramFeature, FuckFeature, TFFeature, BigramFeature
from features.features_unigram import MinAvgMaxEmbeddingFeature
from stst import config
from main_tools import *


train_instances = stst.load_parse_data(train_file, nlp)
dev_instances = stst.load_parse_data(dev_file, nlp)
test_instances = stst.load_parse_data(test_file, nlp)

def average_words(instances):
    word_count = 0.0
    char_count = 0.0
    for instance in instances:
        words = instance.get_sent('word')
        chars = instance.get_sent('char')

        word_count += len(words)
        char_count += len(chars)

    print(word_count / len(instances))

    print(char_count / len(instances))


# Define Model
lr = stst.Classifier(stst.LIB_LINEAR_LR())
svm = stst.Classifier(stst.skLearn_svm())
xgb = stst.Classifier(stst.XGBOOST_prob())
boosting = stst.Classifier(stst.sklearn_GradientBoosting())

model = stst.Model('S-lr-expand', lr)

# model.add(FuckFeature(load=True))
model.add(TFFeature(type='word', convey='count', load=False))
model.add(TFFeature(type='char', convey='count', load=False))
model.add(BigramFeature(type='word', convey='count', load=False))
model.add(BigramFeature(type='char', convey='count', load=False))

headlines_vec = config.EMB_WORD_DIR + '/headlines.vec'

model.add(MinAvgMaxEmbeddingFeature('headlines', 100, headlines_vec, pooling_type='avg', load=False))
model.add(MinAvgMaxEmbeddingFeature('headlines', 100, headlines_vec, pooling_type='all', load=False))


# feature_alation(model)
train_nlpcc(model)
# dev_nlpcc(model)
test_nlpcc(model)

if __name__ == '__main__':
    pass
    # average_words(train_instances)
    # average_words(dev_instances)
    # average_words(test_instances)

    # main()
    # dev_nlpcc(xgb_model_best)



