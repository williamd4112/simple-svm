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

    def get_support_vectors(self):
        return self.model.support_vectors_

class CSVM(SVM):
    def validate(self, X, T, kernel, deg, params, fold=3):
        svc = svm.SVC(kernel=kernel, degree=deg)
       
        for C in params:
            svc.C = C
            score = np.mean(cross_val_score(svc, X, T, n_jobs=8))
            training_error_rate = 1.0 - svc.fit(X, T).score(X, T)
            n_outliers = float(len(X)) * training_error_rate
            logging.info('C = %f; Accuracy = %f, # SV = %d, # Outlier = %d' % (C, score, svc.n_support_.sum(), n_outliers))
            #print('%f,%f,%d,%d' % (C, score, svc.n_support_.sum(), n_outliers))

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
            training_error_rate = 1.0 - svc.fit(X, T).score(X, T)
            n_outliers = float(len(X)) * training_error_rate
            logging.info('NU = %f; Accuracy = %f, # SV = %d, # Outlier = %d' % (nu, score, svc.n_support_.sum(), n_outliers))
            #print('%f,%f,%d,%d' % (nu, score, svc.n_support_.sum(), n_outliers))


    def train(self, X, T, kernel, deg, param):
        svc = svm.NuSVC(nu=param, kernel=kernel, degree=deg)
        svc.fit(X, T)

        self.model = svc
 
