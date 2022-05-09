from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import pandas as pd

def DT_train_model():
    #read the csv files and train model
    data = pd.read_csv('train.csv').values
    clf = DecisionTreeClassifier()
    xtrain = data[0:21000, 1:]
    train_label = data[0:21000, 0]
    clf.fit(xtrain, train_label)
    return clf

def PER_lin_train_model():
    #read the csv files and train model
    data = pd.read_csv('train.csv').values
    clf = Perceptron(tol=1e-3, random_state=0)
    xtrain = data[0:21000, 1:]
    train_label = data[0:21000, 0]
    clf.fit(xtrain, train_label)
    return clf

def PER_nn_train_model():
	#read the csv files and train model
    data = pd.read_csv('train.csv').values
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    xtrain = data[0:21000, 1:]
    train_label = data[0:21000, 0]
    clf.fit(xtrain, train_label)
    return clf