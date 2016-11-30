# coding: utf8
import pandas as pd
from sklearn import neighbors

def nearestNeighbors(pollutant_train_data, pollutant_test_data, n_neighbors):
    X_train = pollutant_train_data.drop('TARGET', axis=1).drop('ID', axis=1)
    Y_train = pollutant_train_data['TARGET']
    X_test = pollutant_test_data.drop('ID', axis=1)

    # paramétrage de la méthode plus proche voisins
    weights = 'distance' # valeurs possibles 'distance' ou 'uniform
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

    # methode plus proche voisins
    fittedKnn = knn.fit(X_train, Y_train)
    Y_test = pd.DataFrame(fittedKnn.predict(X_test))
    Y_test.columns = ['TARGET']
    return pd.concat([pollutant_test_data['ID'], Y_test], axis=1)
