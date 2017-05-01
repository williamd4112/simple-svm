# Introduction
This project is implementation of **C-SVM** and **𝛎-SVM** for multi-class classification. (*see Pattern Recognition and Machine Learning, Bishop 2006*) All datas are processed with **Principle Component Analysis (PCA)** before visualization.

# Dependencies
- numpy v1.12
- sklearn

# Dataset
The MNIST database of handwritten digits  
Reference : http://yann.lecun.com/exdb/mnist/

# Results
Best error rate of each model

|C-SVM   |  𝛎-SVM |
|:-:|:-:|
|0.0224   |  0.0288 |

# Usage
```
usage: main.py [-h] [--train_X TRAIN_X] [--train_T TRAIN_T] [--test_X TEST_X]
               [--test_T TEST_T] [--task {validate,train,eval,plot}]
               [--model {c-svm,nu-svm}] [--kernel {linear,poly,rbf}]
               [--deg DEG] [--c C] [--min_c MIN_C] [--max_c MAX_C]
               [--step_c STEP_C] [--nu NU] [--min_nu MIN_NU] [--max_nu MAX_NU]
               [--step_nu STEP_NU]

optional arguments:
  -h, --help            show this help message and exit
  --train_X TRAIN_X     training data X
  --train_T TRAIN_T     training data T
  --test_X TEST_X       testing data X
  --test_T TEST_T       testing data T
  --task {validate,train,eval,plot}
                        task type
  --model {c-svm,nu-svm}
                        model type
  --kernel {linear,poly,rbf}
                        kernel type
  --deg DEG             degree
  --c C                 c
  --min_c MIN_C         min_c
  --max_c MAX_C         max_c
  --step_c STEP_C       step_c
  --nu NU               nu
  --min_nu MIN_NU       min_nu
  --max_nu MAX_NU       max_nu
  --step_nu STEP_NU     step_nu
```

## To train the model for example
```
python main.py --task train --model nu-svm --kernel linear --nu 0.5 --train_X data/X_train.csv --train_T data/T_train.csv --test_X data/X_test.csv --test_T data/T_test.csv
```
## To evaluate the model for example
```
python main.py --task eval --model nu-svm --train_X data/X_train.csv --train_T data/T_train.csv --test_X data/X_test.csv --test_T data/T_test.csv
```
## To plot the model for example
```
python main.py --task plot --model nu-svm --train_X data/X_train.csv --train_T data/T_train.csv --test_X data/X_test.csv --test_T data/T_test.csv
```
