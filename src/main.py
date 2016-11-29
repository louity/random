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

# methode plus proches voisins

def nearestNeighbour(pollutant, n_neighbors, train_data, test_data):
    # preprocessing
    # on recupere les valeurs du polluant, on enleve les colonne qui n'ont que Nan, on remplace les NaN par la moyenne, on separe X et Y et on enleve l'ID
    train = train_data.loc[train_data['pollutant'] == pollutant].reset_index(drop = True).drop('pollutant', axis = 1).dropna(axis = 1, how = 'all');
    fill_NaN = Imputer(missing_values = np.nan);
    train_fill = pd.DataFrame(fill_NaN.fit_transform(train));
    train_fill.columns = train.columns;

    train_X_preproc = train_fill.drop('TARGET', axis = 1).drop('ID', axis = 1);
    train_Y_preproc = train_fill['TARGET']

    # idem pour les données de test
    test_X = test_data.loc[test_data['pollutant'] == pollutant].reset_index(drop = True).drop('pollutant', axis = 1).dropna(axis = 1, how = 'all');

    test_X_fill = pd.DataFrame(fill_NaN.fit_transform(test_X));
    test_X_fill.columns = test_X.columns;
    test_X_preproc = test_X_fill.drop('ID', axis = 1);


    # paramétrage de la méthode plus proche voisins
    weights = 'uniform' # autre valeur possible 'distance'
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights = weights)

    # methode plus proche voisins
    fittedKnn = knn.fit(train_X_preproc, train_Y_preproc);
    test_Y_predicted = pd.DataFrame(fittedKnn.predict(test_X_preproc));
    test_Y_predicted.columns = ['TARGET'];
    return pd.concat([test_X['ID'], test_Y_predicted], axis = 1);

# methode plus proche voisin polluant par polluant

Y_test_NO2 = nearestNeighbour('NO2', 3, train, X_test)
Y_test_PM10 = nearestNeighbour('PM10', 3, train, X_test)
Y_test_PM2_5 = nearestNeighbour('PM2_5', 3, train, X_test)
Y_test = pd.concat([Y_test_NO2, Y_test_PM10, Y_test_PM2_5]);

Y_test.to_csv('data/Y_test.csv', index = False);
