# coding:utf-8
from __future__ import print_function

import pickle
from numpy import shape
import numpy as np
import os
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn import svm
from sklearn.cross_validation import KFold
import xgboost as xgb

from stst import utils

# __all__ = [
#     "Strategy",
#     "Classifier",
#     "RandomForestRegression",
#     "GradientBoostingRegression",
#     "AverageEnsemble",
#     "LIB_LINEAR_LR",
#     "skLearn_svm",
#     "XGBOOST"
# ]


class Strategy(object):
    def train_model(self, train_file_path, model_path):
        return None

    def test_model(self, test_file_path, model_path, result_file_path):
        return None

    def load_file(self, file_path):
        data = load_svmlight_file(file_path)
        return data[0], data[1]


class Classifier(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def train_model(self, train_file_path, model_path):
        return self.strategy.train_model(train_file_path, model_path)

    def test_model(self, test_file_path, model_path, result_file_path):
        return self.strategy.test_model(test_file_path, model_path, result_file_path)


class RandomForestRegression(Strategy):
    """
    RandomForest Regression

    """
    def __init__(self, n_estimators=300):
        self.trainer = "RandomForest Regression"
        print("==> Using %s Classifier" % (self.trainer))
        self.n_estimators = n_estimators

    def train_model(self, train_file_path, model_path):
        print("==> Load the data ...")
        X_train, Y_train = self.load_file(train_file_path)
        print(train_file_path, shape(X_train))

        print("==> Train the model ...")
        min_max_scaler = preprocessing.MaxAbsScaler()
        X_train_minmax = min_max_scaler.fit_transform(X_train)
        clf = RandomForestRegressor(n_estimators=self.n_estimators)
        clf.fit(X_train_minmax.toarray(), Y_train)

        print("==> Save the model ...")
        pickle.dump(clf, open(model_path, 'wb'))

        scaler_path = model_path.replace('.pkl', '.scaler.pkl')
        pickle.dump(min_max_scaler, open(scaler_path, 'wb'))
        return clf

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the data ...")
        X_test, Y_test = self.load_file(test_file_path)
        print(test_file_path, shape(X_test))

        print("==> Load the model ...")
        clf = pickle.load(open(model_path, 'rb'))
        scaler_path = model_path.replace('.pkl', '.scaler.pkl')
        min_max_scaler = pickle.load(open(scaler_path, 'rb'))

        print("==> Test the model ...")
        X_test_minmax = min_max_scaler.transform(X_test)
        y_pred = clf.predict(X_test_minmax.toarray())

        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred


class GradientBoostingRegression(Strategy):
    """
    Gradient Boosting Regression
    """
    def __init__(self, n_estimators=140):
        self.trainer = "GradientBoostingRegression"
        print("Using %s Classifier" % (self.trainer))
        self.n_estimators = n_estimators

    def train_model(self, train_file_path, model_path):
        print("==> Load the data ...")
        X_train, Y_train = self.load_file(train_file_path)
        print(train_file_path, shape(X_train))

        print("==> Train the model ...")
        min_max_scaler = preprocessing.MaxAbsScaler()
        X_train_minmax = min_max_scaler.fit_transform(X_train)

        clf = GradientBoostingRegressor(n_estimators=self.n_estimators)
        clf.fit(X_train_minmax.toarray(), Y_train)

        print("==> Save the model ...")
        pickle.dump(clf, open(model_path, 'wb'))

        scaler_path = model_path.replace('.pkl', '.scaler.pkl')
        pickle.dump(min_max_scaler, open(scaler_path, 'wb'))
        return clf

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the data ...")
        X_test, Y_test = self.load_file(test_file_path)
        print(test_file_path, shape(X_test))

        print("==> Load the model ...")
        clf = pickle.load(open(model_path, 'rb'))
        scaler_path = model_path.replace('.pkl', '.scaler.pkl')
        min_max_scaler = pickle.load(open(scaler_path, 'rb'))

        print("==> Test the model ...")
        X_test_minmax = min_max_scaler.transform(X_test)
        y_pred = clf.predict(X_test_minmax.toarray())

        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred


class AverageEnsemble(Strategy):
    def __init__(self):
        self.trainer = "Average Ensemble"
        print("Using %s Classifier" % (self.trainer))


    def train_model(self, train_file_path, model_path):
        pass

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the data ...")
        X_test, Y_test = self.load_file(test_file_path)
        print(test_file_path, shape(X_test))
        X_test = X_test.toarray()
        for x in X_test[:10]:
            print(x)

        print("==> Test the model ...")
        y_pred = []
        for x in X_test:
            x = sum(x) / len(x)
            y_pred.append(x)

        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred



''' LibLinear '''
class LIB_LINEAR_LR(Strategy):
    def __init__(self):
        self.trainer = "LIB_LINEAR LR"
        self.LIB_LINEAR_PATH = '/opt/liblinear-multicore-2.11-1'
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):

         # - s
         # 0 -- L2-regularized logistic regression (primal)
         # 1 -- L2-regularized L2-loss support vector classification (dual)
         # 2 -- L2-regularized L2-loss support vector classification (primal)
         # 3 -- L2-regularized L1-loss support vector classification (dual)
         # 4 -- support vector classification by Crammer and Singer
         # 5 -- L1-regularized L2-loss support vector classification
         # 6 -- L1-regularized logistic regression

         # -c cost
         # -b probability_estimates: whether to output probability estimates, 0 or 1     (default 0); currently for logistic regression only

         # -n nr_thread
        print("==> Train the model ...")
        cmd = self.LIB_LINEAR_PATH + \
              "/python3 -n 8 -s 0 -c 1 %s %s > /dev/null" % (train_file_path, model_path)
        # print(cmd)
        os.system(cmd)

        return model_path

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Test the model ...")
        cmd = self.LIB_LINEAR_PATH + "/predict " + test_file_path + " " + model_path + " " + result_file_path
        # print(cmd)
        os.system(cmd)

        cmd = self.LIB_LINEAR_PATH + "/predict -b 1 " + test_file_path + " " + model_path + " " + result_file_path + '.prob'
        os.system(cmd)

        y_pred = open(result_file_path, 'r').readlines()
        y_pred = [eval(x) for x in y_pred]
        return y_pred


class skLearn_svm(Strategy):
    def __init__(self):
        self.trainer = "skLearn svm"
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = self.load_file(train_file_path)

        clf = svm.LinearSVC()

        print("==> Train the model ...")
        clf.fit(train_X, train_y)

        print("==> Save the model ...")
        pickle.dump(clf, open(model_path, 'wb'))

        return clf

    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = self.load_file(test_file_path)

        print("==> Load the model ...")
        clf = pickle.load(open(model_path, 'rb'))

        y_pred = clf.predict(test_X)

        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)

        return y_pred


class sklearn_GradientBoosting(Strategy):
    """
    Gradient Boosting Regression
    """
    def __init__(self, n_estimators=140):
        self.trainer = "sklearn_GradientBoosting"
        print("Using %s Classifier" % (self.trainer))
        self.n_estimators = n_estimators

    def train_model(self, train_file_path, model_path):
        print("==> Load the data ...")
        train_X, train_y = self.load_file(train_file_path)

        print("==> Train the model ...")
        clf = GradientBoostingClassifier(n_estimators=self.n_estimators)
        clf.fit(train_X, train_y)

        print("==> Save the model ...")
        pickle.dump(clf, open(model_path, 'wb'))

        return clf

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the data ...")
        test_X, test_y = self.load_file(test_file_path)

        print("==> Load the model ...")
        clf = pickle.load(open(model_path, 'rb'))

        print("==> Test the model ...")
        y_pred = clf.predict(test_X.toarray())

        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred


class XGBOOST(Strategy):
    def __init__(self):
        self.trainer = "XGBOOST"
        print("Using %s Classifier" % (self.trainer))

        # specify parameters via map
        # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
        # param = { 'silent': 1, 'eta': 0.1, 'max_depth': 10, "booster" : "gbtree",}

        # param = {'silent': 1, "booster":"gbtree",  "subsample": 0.9, "colsample_bytree": 0.7, "seed":1301 }
        # param = {'silent':1, 'objective':'multi:softmax', 'num_class':5}

        param = {'objective': 'multi:softmax', 'num_class': 18, 'booster': 'gbtree',
                 'max_depth': 5
                 }
        self.param  = param

    def train_model(self, train_file_path, model_path):
        print("==> Train the model ...")
        # read in data
        dtrain = xgb.DMatrix(train_file_path)

        num_round = 30

        bst = xgb.train(self.param, dtrain, num_round)
        # make prediction
        print("==> Save the model ...")
        bst.save_model(model_path)
        # pickle.dump(bst, open(model_path, 'wb'))
        return bst

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the model ...")
        # bst = pickle.load(open(model_path, 'rb'))

        bst = xgb.Booster(self.param)
        bst.load_model(model_path)

        print("==> Test the model ...")
        dtest = xgb.DMatrix(test_file_path)
        y_pred = bst.predict(dtest)
        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred

class XGBOOST_prob(Strategy):
    def __init__(self):
        self.trainer = "XGBOOST"
        print("Using %s Classifier" % (self.trainer))

        # specify parameters via map
        # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
        # param = { 'silent': 1, 'eta': 0.1, 'max_depth': 10, "booster" : "gbtree",}

        # param = {'silent': 1, "booster":"gbtree",  "subsample": 0.9, "colsample_bytree": 0.7, "seed":1301 }
        # param = {'silent':1, 'objective':'multi:softmax', 'num_class':5}

        param = {'objective': 'multi:softprob', 'num_class': 18, 'booster': 'gbtree',
                 'max_depth': 5, 'sub_sample':0.8,
                 }
        self.param = param

    def train_model(self, train_file_path, model_path):
        print("==> Train the model ...")
        # read in data
        dtrain = xgb.DMatrix(train_file_path)

        num_round = 100

        bst = xgb.train(self.param, dtrain, num_round)
        # make prediction
        print("==> Save the model ...")
        bst.save_model(model_path)
        # pickle.dump(bst, open(model_path, 'wb'))
        return bst

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the model ...")
        # bst = pickle.load(open(model_path, 'rb'))

        bst = xgb.Booster(self.param)
        bst.load_model(model_path)

        print("==> Test the model ...")
        dtest = xgb.DMatrix(test_file_path)
        y_probs = bst.predict(dtest).reshape(-1, 18)

        with open(result_file_path+'.pkl', 'wb') as f:
            pickle.dump(y_probs, f)

        y_pred = np.argmax(y_probs, axis=1)
        print("==> Save the result ...")
        with utils.create_write_file(result_file_path) as f:
            for y in y_pred:
                print(y, file=f)
        return y_pred
