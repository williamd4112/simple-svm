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

# Visualization (Decision boundary)
|C-SVM   |  𝛎-SVM |
| ------------- |:------------:|
|![csvm](/doc/bound_gen.png)|![nuvm](/doc/bound_dis_lda.png)|

# To train the model (examples)
Training scripts use default training data in data/class*.npy and default training hyperparameters. If you want to use your own data, please see the manual of main.py
```
./train_generative.sh {model output path}
./train_dicriminative.sh {model output path}
./train_dicriminative_lda.sh {model output path}
```

# To validate the model (examples)
```
./validate_generative.sh
./validate_dicriminative.sh 
./validate_dicriminative_lda.sh 
```

# To test the model (examples)
```
./test.sh {model input} {result output} {testing data} {model type [dis|gen]}

e.g.
./test.sh model/model-dis data/class1.npy,data/class2.npy,data/class3.npy dis
```
