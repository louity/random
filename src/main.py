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

# lire les données
X_train = pd.read_csv('data/X_train.csv')
Y_train = pd.read_csv('data/Y_train.csv')
X_test = pd.read_csv('data/X_test.csv');

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

# methode plus proches voisins pour le NO2

# preprocessing
# on recupere les valeurs du NO2
train_NO2 = train.loc[train['pollutant'] == 'NO2'].reset_index(drop = True).drop('pollutant', axis = 1);
# on complete les valeurs non renseignées (nécessaire pour l'algo)
fill_NaN = Imputer(missing_values = np.nan);
train_NO2_fill = pd.DataFrame(fill_NaN.fit_transform(train_NO2));
train_NO2_fill.columns = train_NO2.columns;

# idem pour les données de test
X_test_NO2 = X_test.loc[X_test['pollutant'] == 'NO2'].reset_index(drop = True).drop('pollutant', axis = 1);
X_test_NO2_fill = pd.DataFrame(fill_NaN.fit_transform(X_test_NO2));
X_test_NO2_fill.columns = X_test_NO2.columns;


# paramétrage de la méthode plus proche voisins
n_neighbors = 5
weights  = 'uniform' # autre valeur possible 'distance'
knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

X = train_NO2_fill.drop('TARGET', axis = 1).drop('ID', axis = 1);
Y = train_NO2_fill['TARGET']

fittedKnn = knn.fit(X, Y);
Y_predicted = pd.DataFrame(fittedKnn.predict(X_test_NO2_fill.drop('ID', axis = 1)));
Y_predicted.columns = ['TARGET'];
Y_test_NO2 = pd.concat([X_test_NO2['ID'], Y_predicted], axis = 1);

# ecrire le résultat dans un fichier csv
Y_test_NO2.to_csv('data/Y_test.csv', index = False);
