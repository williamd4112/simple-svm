import logging
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

import cPickle as pickle

class SVM(object): 
    def save(self, path):
        pickle.dump(self.model, open(path, "wb"), True) 

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))

    def validate(self, X, T, kernel, deg, params, fold=3):
        raise NotImplemented()

    def train(self, X, T, kernel, deg, param):
        raise NotImplemented()

    def test(self, X):
        return self.model.predict(X)

    def eval(self, X, T):
        y = self.test(X).astype(np.int32)
        t = T.astype(np.int32)
        return float(np.equal(y, t).sum()) / len(t)

class CSVM(SVM):
    def validate(self, X, T, kernel, deg, params, fold=3):
        svc = svm.SVC(kernel=kernel, degree=deg)
       
        for C in params:
            svc.C = C
            score = np.mean(cross_val_score(svc, X, T, n_jobs=8))
            logging.info('C = %d; Accuracy = %f' % (C, score))

    def train(self, X, T, kernel, deg, param):
        svc = svm.SVC(C=param, kernel=kernel, degree=deg)
        svc.fit(X, T)

        self.model = svc

class NuSVM(SVM):
    def validate(self, X, T, kernel, deg, params, fold=3):
        svc = svm.NuSVC(kernel=kernel, degree=deg)
       
        for nu in params:
            svc.nu = nu
            score = np.mean(cross_val_score(svc, X, T, n_jobs=8))
            logging.info('Nu = %f; Accuracy = %f' % (nu, score))

    def train(self, X, T, kernel, deg, param):
        svc = svm.NuSVC(nu=param, kernel=kernel, degree=deg)
        svc.fit(X, T)

        self.model = svc
 
