import argparse
import sys
import logging
import csv
import numpy as np

from model_np import *
from plot import *

from sklearn.decomposition import PCA

def load_csv(path):
    csv_file = open(path, 'rb')
    csv_reader = csv.reader(csv_file, delimiter=',')
    datas = np.array([data for data in csv_reader]).astype(np.float32)
    return datas

def get_model(args):
    logging.info('Model = %s' % args.model)
    if args.model == 'c-svm':
        return CSVM()
    else:
        return NuSVM()

def get_cross_validation_params(args):
    if args.model == 'c-svm':
        return np.arange(args.min_c, args.max_c, args.step_c)
    else:
        return np.arange(args.min_nu, args.max_nu, args.step_nu)

def get_param(args):
    if args.model == 'c-svm':
        return args.c
    else:
        return args.nu

def get_kernel(args):
    logging.info('Kernel = %s, Degree = %d' % (args.kernel, args.deg))
    return args.kernel

def main(args):    
    model = get_model(args)
    if args.task == 'validate':
        X_Train = load_csv(args.train_X)
        T_Train = load_csv(args.train_T).flatten()
        logging.info('Validation')
        model.validate(X_Train, T_Train, kernel=get_kernel(args), deg=args.deg, params=get_cross_validation_params(args))
    elif args.task == 'train':
        X_Train = load_csv(args.train_X)
        T_Train = load_csv(args.train_T).flatten()
        logging.info('Training')
        model.train(X_Train, T_Train, kernel=get_kernel(args), deg=args.deg, param=get_param(args))
        model.save('%s-model' % args.model)
        logging.info('Model saved at %s-model' % args.model)
    elif args.task == 'eval':
        X_Test = load_csv(args.test_X)
        T_Test = load_csv(args.test_T).flatten()
        model.load('%s-model' % args.model)
        logging.info('Model loaded from %s-model' % args.model)
        logging.info('Evaluating')
        acc = model.eval(X_Test, T_Test)
        logging.info('Evaluting accuracy = %f' % acc)
    elif args.task == 'plot':
        X_Train = load_csv(args.train_X)
        T_Train = load_csv(args.train_T).flatten()
        X_Test = load_csv(args.test_X)
        T_Test = load_csv(args.test_T).flatten()

        model.load('%s-model' % args.model)
        logging.info('Model loaded from %s-model' % args.model)
        logging.info('Plotting')
          
        pca = PCA(n_components=2)
        pca.fit(X_Train)

        X_Support = model.get_support_vectors()     
        X_Support_pca = pca.transform(X_Support)
 
        def predict(x):
            x_inv = pca.inverse_transform(x)
            return model.test(x_inv)

        Y_Support = predict(X_Support_pca)
        plot_decision_boundary(predict, X_Support_pca, Y_Support, 0.05) 
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_X', help='training data X', type=str)
    parser.add_argument('--train_T', help='training data T', type=str)
    parser.add_argument('--test_X', help='testing data X', type=str)
    parser.add_argument('--test_T', help='testing data T', type=str)

    parser.add_argument('--task', help='task type', type=str, choices=['validate', 'train', 'eval', 'plot'], default='validate')
    parser.add_argument('--model', help='model type', type=str, choices=['c-svm', 'nu-svm'], default='c-svm')
    parser.add_argument('--kernel', help='kernel type', type=str,  choices=['linear', 'poly', 'rbf'], default='linear')
    parser.add_argument('--deg', help='degree', type=int, default=1)

    parser.add_argument('--c', help='c', type=float, default=0.0)
    parser.add_argument('--min_c', help='min_c', type=float, default=0.0)
    parser.add_argument('--max_c', help='max_c', type=float, default=1.0)
    parser.add_argument('--step_c', help='step_c', type=float, default=0.1)

    parser.add_argument('--nu', help='nu', type=float, default=0.0)
    parser.add_argument('--min_nu', help='min_nu', type=float, default=0.0)
    parser.add_argument('--max_nu', help='max_nu', type=float, default=1.0)
    parser.add_argument('--step_nu', help='step_nu', type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
 
    main(args)
