# coding: utf8
import stst
from  stst import Model
nlp = None

train_file = './data/nlpcc_data/word/train-exp-fix.txt'
dev_file = './data/nlpcc_data/word/dev-exp-fix.txt'
test_file = './data/nlpcc_data/word/test-exp-fix.txt'

# train_file = './data/nlpcc_data/word/train.txt'
# dev_file = './data/nlpcc_data/word/dev.txt'
# test_file = './data/nlpcc_data/word/test.txt'

# test_file = './data/nlpcc_data/word/test-exp-fix.txt'
# train_file = './data/nlpcc_data/word/devaa.txt'
# dev_file = './data/nlpcc_data/word/devab.txt'

def train_nlpcc(model):
    train_instances = stst.load_parse_data(train_file, nlp)
    model.train(train_instances, train_file)


def dev_nlpcc(model):
    dev_instances = stst.load_parse_data(dev_file, nlp)
    model.test(dev_instances, dev_file)
    # evaluation
    acc, _, _, _ = stst.Evaluation(dev_file, model.output_file)
    print(acc)
    return acc


def test_nlpcc(model):
    test_instances = stst.load_parse_data(test_file, nlp)
    model.test(test_instances, test_file)
    acc, _, _, _ = stst.Evaluation(test_file, model.output_file)
    print(acc)
    return acc


def stack_nlpcc(model):
    if 'stack' not in model.model_name:
        raise NotImplementedError
    train_instances = stst.load_parse_data(train_file, nlp)
    model.cross_validation(train_instances, train_file)
    # evaluation
    acc, _, _, _ = stst.Evaluation(train_file, model.output_file)
    print(acc)
    return acc


def hill_climbing(model, choose_list=[]):
    chooses = choose_list
    feature_list= model.feature_list
    visited = [True if x in choose_list else False for x in range(len(feature_list))]

    for idx in range(len(choose_list), len(feature_list)):
        chooseIndex = -1
        best_score = 0.0
        best_test_score = 0.0
        chooses.append(-1)
        for i in range(len(feature_list)):
            if visited[i] == False:
                chooses[idx] = i
                feature = [feature_list[s] for s in chooses]
                # print(len(feature_list))
                model.feature_list = feature
                train_nlpcc(model)
                cur_score = dev_nlpcc(model)
                test_score = test_nlpcc(model)
                stst.record('./data/records.csv', cur_score, test_score, model)
                if best_score < cur_score:
                    chooseIndex = i
                    best_score = cur_score
                    best_test_score = test_score

        chooses[idx] = chooseIndex
        visited[chooseIndex] = True
        # feature = [ feature_list[s] for s in chooses]
        print('Best Score: %.2f %%,  %.2f%%,choose Feature %s' % (best_score * 100, best_test_score * 100,
                                                                  feature_list[chooseIndex].feature_name))



def feature_alation(model):
    train_instances = stst.load_parse_data(train_file)
    dev_instances = stst.load_parse_data(dev_file)

    feature_list= model.feature_list

    model.train(train_instances, train_file)
    model.test(dev_instances, dev_file)
    exit(1)

    for feature in feature_list:
        model.feature_list = [feature]
        model.train(train_instances, train_file)
        model.test(dev_instances, dev_file)
        # evaluation
        acc, _, _, _ = stst.Evaluation(dev_file, model.output_file)
        print(feature.feature_name)
        print(acc)

if __name__ == '__main__':
    pass