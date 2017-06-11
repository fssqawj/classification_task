# coding: utf8
import stst
from stst import Model

nlp = None

train_file = './data/nlpcc_data/word/train-exp-fix.txt'
dev_file = './data/nlpcc_data/word/dev-exp-fix.txt'
test_file = './data/nlpcc_data/word/test-exp-fix.txt'


# Define Model
lr = stst.Classifier(stst.LIB_LINEAR_LR())

model = stst.Model('S-lr', lr)

train_instances = stst.load_parse_data(train_file, nlp)
model.train(train_instances, train_file)


dev_instances = stst.load_parse_data(dev_file, nlp)
model.test(dev_instances, dev_file)

# evaluation
acc, _, _, _ = stst.Evaluation(dev_file, model.output_file)
print(acc)
