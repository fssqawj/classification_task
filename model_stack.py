from __future__ import print_function
import stst
from features.features_unigram import UnigramFeature, FuckFeature, TFFeature, BigramFeature
from features.features_unigram import MinAvgMaxEmbeddingFeature
from stst import config
from main_tools import *
# Define Model
lr = stst.Classifier(stst.LIB_LINEAR_LR())
svm = stst.Classifier(stst.skLearn_svm())
xgb = stst.Classifier(stst.XGBOOST_prob())
boosting = stst.Classifier(stst.sklearn_GradientBoosting())


def get_stack_model(model):
    model_stack = Model(model.model_name + '-stack', model.classifier)
    model_stack.feature_list = model.feature_list
    return model_stack


def stack_model(is_training, model):
    model_stack = get_stack_model(model)
    if is_training:
        stack_nlpcc(model_stack)
    return model_stack


def train_model_best(is_training=False, model_name='S2-gb', classifier=lr):
    model_best = stst.Model(model_name, classifier)
    model_best.add(TFFeature(type='word', convey='count', load=True))
    model_best.add(TFFeature(type='char', convey='count', load=True))
    model_best.add(BigramFeature(type='word', convey='count', load=True))
    model_best.add(BigramFeature(type='char', convey='count', load=True))

    emb_wd_50_file =  config.EMB_WORD_DIR + '/embedding.50'
    emb_wd_100_file =  config.EMB_WORD_DIR + '/embedding.100'
    emb_wd_200_file =  config.EMB_WORD_DIR + '/embedding.200'
    emb_wd_300_file =  config.EMB_WORD_DIR + '/embedding.300'

    headlines_vec = config.EMB_WORD_DIR + '/headlines.vec'

    model_best.add(MinAvgMaxEmbeddingFeature('headlines', 100, headlines_vec, pooling_type='avg', load=True))
    model_best.add(MinAvgMaxEmbeddingFeature('headlines', 100, headlines_vec, pooling_type='all', load=True))

    if is_training:
        train_nlpcc(model_best)
        dev_nlpcc(model_best)
    return model_best


model_best = train_model_best(False)
model_best_stack = stack_model(False, model_best)

xgb_model_best = train_model_best(False, 'S2-xgb', xgb)
xgb_model_best_stack = stack_model(False, xgb_model_best)


def train_model_emb(is_training=False):
    model_emb = stst.Model('S3-gb', lr)
    headlines_vec = config.EMB_WORD_DIR + '/headlines.vec'
    model_emb.add(MinAvgMaxEmbeddingFeature('headlines', 100, headlines_vec, pooling_type='avg', load=True))
    model_emb.add(MinAvgMaxEmbeddingFeature('headlines', 100, headlines_vec, pooling_type='all', load=True))
    if is_training:
        train_nlpcc(model_emb)
        dev_nlpcc(model_emb)
    return model_emb

model_emb =  train_model_emb(False)
model_emb_stack = stack_model(False, model_emb)


def stack():
    model_stack = stst.Model('Stack1', boosting)

    model_stack.add(model_best_stack)
    model_stack.add(xgb_model_best_stack)
    model_stack.add(model_emb_stack)

    train_nlpcc(model_stack)

    model_stack.feature_list = []
    model_stack.add(model_best)
    model_stack.add(model_emb)
    model_stack.add(xgb_model_best)

    dev_nlpcc(model_stack)


if __name__ == '__main__':
    pass
    stack()
