# coding: utf8
import pandas as pd
from sklearn import neighbors
import preprocessing
import loadData

def nearest_neighbors(train_data, test_data, n_neighbors):
    X_train = train_data.drop('TARGET', axis=1).drop('ID', axis=1)
    Y_train = train_data['TARGET']
    X_test = test_data.drop('ID', axis=1)

    # paramétrage de la méthode plus proche voisins
    weights = 'distance' # valeurs possibles 'distance' ou 'uniform'
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

    # methode plus proche voisins
    fittedKnn = knn.fit(X_train, Y_train)
    Y_test = pd.DataFrame(fittedKnn.predict(X_test))
    Y_test.columns = ['TARGET']
    return pd.concat([test_data['ID'], Y_test], axis=1)

def nearest_neighbors_by_pollutant(train_data=loadData.XY_train, test_data=loadData.X_test):
    pollutant_train_datas = preprocessing.separatePollutantDatas(train_data, True)
    pollutant_test_datas = preprocessing.separatePollutantDatas(test_data, True)
    n_neighbors = 4
    Y_test = pd.DataFrame();

    for pollutant in pollutant_train_datas:
        print 'nearest neighbors for ', pollutant , '...'
        train = pollutant_train_datas[pollutant]
        test = pollutant_test_datas[pollutant]
        Y_test_pollutant = nearest_neighbors(train, test, n_neighbors)
        Y_test = pd.concat([Y_test, Y_test_pollutant])
        print 'done'
    filename = 'data/Y_test_' + str(n_neighbors) + 'nn_by_pollutant.csv'
    Y_test.to_csv(filename, index=False)

def nearest_neighbors_by_pollutant_and_zone(train_data=loadData.XY_train, test_data=loadData.X_test):
    pollutant_zone_train_datas = preprocessing.separatePollutantAndZoneDatas(train_data, True)
    pollutant_zone_test_datas = preprocessing.separatePollutantAndZoneDatas(test_data, True)
    n_neighbors = 4
    Y_test = pd.DataFrame();

    for pollutant_key in pollutant_zone_train_datas:
        zone_train_datas = pollutant_zone_train_datas[pollutant_key]
        zone_test_datas = pollutant_zone_test_datas[pollutant_key]
        for zone_id_key in zone_train_datas:
            print 'nearest neighbors for ', pollutant_key , ' in zone ', zone_id_key, '...'
            train = zone_train_datas[zone_id_key]
            test = zone_test_datas[zone_id_key]
            if set(train.columns) != set(test.columns):
                print 'columns are not the same for train and test, zone ', zone_id_key, ' pollutant ', pollutant_key
            Y_test_pollutant_zone = nearest_neighbors(train, test, n_neighbors)
            Y_test = pd.concat([Y_test, Y_test_pollutant_zone])
        print 'done'
    filename = 'data/Y_test_' + str(n_neighbors) + 'nn_by_pollutant_and_zone.csv'
    Y_test.to_csv(filename, index=False)
