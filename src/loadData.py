# coding: utf8
import pandas as pd

# lire les données
X_train = pd.read_csv('data/X_train.csv')
Y_train = pd.read_csv('data/Y_train.csv')
X_test = pd.read_csv('data/X_test.csv')

# fusionner X et Y en un seul dataframe
XY_train = pd.concat([X_train, Y_train['TARGET']], axis=1)
