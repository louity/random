# coding: utf8
""" packages
    Le package panda definit des dataframe avec plein de fonctions pratiques
    voir http://pandas.pydata.org/
"""
import numpy as np
import pandas as pd

# scripts
import plotUtils

# lire les données
X_train = pd.read_csv('data/X_train.csv')
Y_train = pd.read_csv('data/Y_train.csv')

# fusionner X et Y and un seul dataframe dans un seul dataframe
train = pd.concat([X_train, Y_train['TARGET']], axis = 1)

# repérer les noms des différents polluants
pollutants = set(train['pollutant']);
print(pollutants);

# récupérer les differentes valeurs des polluants
NO2values = train.loc[train['pollutant'] == 'NO2']['TARGET'].values
PM10values = train.loc[train['pollutant'] == 'PM10']['TARGET'].values
PM2_5values = train.loc[train['pollutant'] == 'PM2_5']['TARGET'].values

# afficher un histogramme
plotUtils.plot_histogram(NO2values, 'NO2');
