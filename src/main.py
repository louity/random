# coding: utf8
""" packages
    Le package panda definit des dataframe avec plein de fonctions pratiques
    voir http://pandas.pydata.org/
"""
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import Imputer

# scripts
import plotUtils
import nearestNeighbors
import preprocessing
import verification

# lire les données
X_train = pd.read_csv('data/X_train.csv')
Y_train = pd.read_csv('data/Y_train.csv')
X_test = pd.read_csv('data/X_test.csv')

# fusionner X et Y en un seul dataframe
XY_train = pd.concat([X_train, Y_train['TARGET']], axis=1)

# repérer les noms des différents polluants
pollutant_train_datas = preprocessing.separatePollutantDatas(XY_train, True)
pollutant_test_datas = preprocessing.separatePollutantDatas(X_test, True)

# methode plus proche voisin polluant par polluant
n_neighbors = 4
Y_test = pd.DataFrame();

for pollutant in pollutant_train_datas:
    pollutant_train_data = pollutant_train_datas[pollutant]
    pollutant_test_data = pollutant_test_datas[pollutant]
    Y_test_pollutant = nearestNeighbors.nearestNeighbors(pollutant_train_data, pollutant_test_data, n_neighbors)
    Y_test = pd.concat([Y_test, Y_test_pollutant])

Y_test.to_csv('data/Y_test.csv', index=False)
