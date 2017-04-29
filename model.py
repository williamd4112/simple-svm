import logging
import numpy as np

from svmutil import *

class SVM(object):
    def save(self, path):
        svm_save_model(path, self.model)
    
    def load(self, path):
        self.model = svm_load_model(path)

    def validate(self, X, T, kernel, deg, params, fold=3):
        raise NotImplemented()

    def train(self, X, T, kernel, deg, param):
        raise NotImplemented()

class CSVM(SVM):
    def validate(self, X, T, kernel, deg, params, fold=3):
        '''
        param X: [n_samples, n_dimensions]
        param T: [n_samples, label(1 ~ K)]
        '''
        K = int(np.max(T))
        N = len(X)
        c_list = params

        best_c = None
        best_acc = 0.0
        for c in c_list:
            logging.info('C = %f' % c)
            acc = svm_train(T.reshape(N).tolist(), X.tolist(), '-s 0 -t %s -c %d -d %d -v %d -q' % (kernel, c, deg, fold))
            
            if acc > best_acc:
                best_c = c
        return best_c

    def train(self, X, T, kernel, deg, param):
        '''
        param X: [n_samples, n_dimensions]
        param T: [n_samples, label(1 ~ K)]
        '''
        K = int(np.max(T))
        N = len(X)
        
        self.model = svm_train(T.reshape(N).tolist(), X.tolist(), '-s 0 -t %s -c %d -d %d -q' % (kernel, param, deg))

class NuSVM(SVM):
    def validate(self, X, T, kernel, deg, params, fold=3):
        '''
        param X: [n_samples, n_dimensions]
        param T: [n_samples, label(1 ~ K)]
        '''
        K = int(np.max(T))
        N = len(X)

        nu_list = params
        best_nu = None
        best_acc = 0.0
        for nu in nu_list:
            logging.info('NU = %f' % nu)
            acc = svm_train(T, X, '-s 1 -t %s -n %f -d %d -v %d -q' % (kernel, nu, deg, fold))
            
            if acc > best_acc:
                best_nu = nu
        return best_nu

    def train(self, X, T, kernel, deg, param):
        '''
        param X: [n_samples, n_dimensions]
        param T: [n_samples, label(1 ~ K)]
        '''
        K = int(X)
        N = len(X)
        
        self.model = svm_train(T.reshape(N).tolist(), X.tolist(), '-s 1 -t %s -n %f -d %d -q' % (kernel, param, deg))
    
    def test(self, X, T):
        N = len(X)
        return svm_predict(T, X, self.model)        
 
