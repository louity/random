from readData import *
from public_mean_square_error import *


import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


def predictDynamique(dataTrain, dataTest):
    dno2Train, yno2Train = getLearningData(getPollutant(dataTrain, 'NO2'), statiques = False);
    xno2Train = dno2Train.as_matrix();
    dpm10Train, ypm10Train = getLearningData(getPollutant(dataTrain, 'PM10'), statiques = False);
    xpm10Train = dpm10Train.as_matrix()
    dpm2_5Train, ypm2_5Train = getLearningData(getPollutant(dataTrain, 'PM2_5'), statiques = False);
    xpm2_5Train = dpm2_5Train.as_matrix()

    #Training of the gradient boosting
    pno2 = ensemble.GradientBoostingRegressor(**params)
    ppm10 = ensemble.GradientBoostingRegressor(**params)
    ppm2_5 = ensemble.GradientBoostingRegressor(**params)

    pno2.fit(xno2Train, yno2Train);
    ppm10.fit(xpm10Train, ypm10Train);
    ppm2_5.fit(xpm2_5Train, ypm2_5Train);


    #Testing part
    xno2Test, yno2Test = getLearningData(getPollutant(dataTest, 'NO2'), statiques = False);
    xpm10Test, ypm10Test = getLearningData(getPollutant(dataTest, 'PM10'), statiques = False);
    xpm2_5Test, ypm2_5Test = getLearningData(getPollutant(dataTest, 'PM2_5'), statiques = False);
    idno2 = getIdPollutant(dataTest, 'NO2');
    idpm10 = getIdPollutant(dataTest, 'PM10');
    idpm2_5 = getIdPollutant(dataTest, 'PM2_5');

    if(len(xno2Test) > 0):
        yno2Test = pno2.predict(xno2Test);
    if(len(xpm10Test) > 0):
        ypm10Test = ppm10.predict(xpm10Test);
    if(len(xpm2_5Test) > 0):
        ypm2_5Test = ppm2_5.predict(xpm2_5Test);

    nTest = len(dataTest.index);
    yTest = np.zeros(nTest);

    yTest[idno2] = yno2Test[:];
    yTest[idpm10] = ypm10Test[:];
    yTest[idpm2_5] = ypm2_5Test[:];

    print('Check length : {0} == {1} ?'.format(nTest, yno2Test.size + ypm2_5Test.size + ypm10Test.size))

    return arrayToResult(yTest, dataTest);

#Predict using a learner for each zone/pollutant (18 learner)
def predictZones(dataTrain, dataTest, plotFeaturesInfluence = False):

    """ After Test : works fine with pm10 and pm2_5, but not at all with no2 (500 mse)

    Here an example of the ouput :

            Zone 0 :
         - Training :: no2 : 28254 datas, pm10 : 28032 data pm2_5 : 0 data
         - Test :: no2 : 14127 datas, pm10 : 14127 data pm2_5 : 0 data
        No2 : mse : 80.88636053185851
        PM10 : mse : 41.65295502160105
        Zone 1 :
         - Training :: no2 : 13878 datas, pm10 : 13878 data pm2_5 : 24392 data
         - Test :: no2 : 13878 datas, pm10 : 7958 data pm2_5 : 0 data
        No2 : mse : 532.5888800207944
        PM10 : mse : 57.5857369074756
        Zone 2 :
         - Training :: no2 : 13573 datas, pm10 : 13573 data pm2_5 : 13573 data
         - Test :: no2 : 13573 datas, pm10 : 13545 data pm2_5 : 13330 data
        No2 : mse : 412.0405340738566
        PM10 : mse : 81.60739907505635
        PM2_5 : mse : 22.805464325067057
        Zone 3 :
         - Training :: no2 : 28218 datas, pm10 : 27832 data pm2_5 : 0 data
         - Test :: no2 : 14109 datas, pm10 : 0 data pm2_5 : 0 data
        No2 : mse : 410.0624421540782
        Zone 4 :
         - Training :: no2 : 27671 datas, pm10 : 27674 data pm2_5 : 0 data
         - Test :: no2 : 13839 datas, pm10 : 0 data pm2_5 : 0 data
        No2 : mse : 96.59055195329455
        Zone 5 :
         - Training :: no2 : 13268 datas, pm10 : 28124 data pm2_5 : 0 data
         - Test :: no2 : 14090 datas, pm10 : 13653 data pm2_5 : 0 data
        No2 : mse : 148.13373721193545
        PM10 : mse : 163.07683409699294
        MSE: 191.6078

     """

    #initialisation of the results
    nTest = len(dataTest.index);
    yTest = np.zeros(nTest);

    for z in zone_id:

        #Training
        dataTrainZone = getZone(dataTrain, z);

        dno2Train, yno2Train = getLearningZoneData(getPollutant(dataTrainZone, 'NO2'), z);
        xno2Train = dno2Train.as_matrix();
        namesno2 = dno2Train.columns;
        dpm10Train, ypm10Train = getLearningZoneData(getPollutant(dataTrainZone, 'PM10'), z);
        xpm10Train = dpm10Train.as_matrix();
        namespm10 = dpm10Train.columns;
        dpm2_5Train, ypm2_5Train = getLearningZoneData(getPollutant(dataTrainZone, 'PM2_5'), z);
        xpm2_5Train = dpm2_5Train.as_matrix();
        namespm2_5 = dpm2_5Train.columns;

        print("Zone {0} : \n - Training :: no2 : {1} datas, pm10 : {2} data pm2_5 : {3} data".format(z, len(xno2Train), len(xpm10Train), len(xpm2_5Train)))

        #Training of the gradient boosting
        pno2 = ensemble.GradientBoostingRegressor(**params)
        ppm10 = ensemble.GradientBoostingRegressor(**params)
        ppm2_5 = ensemble.GradientBoostingRegressor(**params)

        no2l = False; pm10l = False; pm2_5l = False;

        if(len(xno2Train) > 0):
            pno2.fit(xno2Train, yno2Train);
            no2l = True;
        if(len(xpm10Train) > 0):
            ppm10.fit(xpm10Train, ypm10Train);
            pm10l = True;
        if(len(xpm2_5Train) > 0):
            ppm2_5.fit(xpm2_5Train, ypm2_5Train);
            pm2_5l = True;

        #Prediction
        dataTestZone = getZone(dataTest, z);

        #Testing part
        xno2Test, yno2T = getLearningZoneData(getPollutant(dataTestZone, 'NO2'), z);
        xpm10Test, ypm10T = getLearningZoneData(getPollutant(dataTestZone, 'PM10'), z);
        xpm2_5Test, ypm2_5T = getLearningZoneData(getPollutant(dataTestZone, 'PM2_5'), z);

        print(" - Test :: no2 : {1} datas, pm10 : {2} data pm2_5 : {3} data".format(z, len(xno2Test), len(xpm10Test), len(xpm2_5Test)))

        idno2 = getIdPollutant(dataTestZone, 'NO2');
        idpm10 = getIdPollutant(dataTestZone, 'PM10');
        idpm2_5 = getIdPollutant(dataTestZone, 'PM2_5');

        yno2Test = np.zeros(0); ypm10Test = np.zeros(0); ypm2_5Test = np.zeros(0);

        if(len(xno2Test) > 0):
            if(not no2l):
                print("Error : no training data for pollutant NO2 in zone {0}".format(z))
            yno2Test = pno2.predict(xno2Test);
            yTest[idno2] = yno2Test[:];
            print("No2 : mse : {0}".format(score_function(yno2Test, yno2T)))
            if(plotFeaturesInfluence):
                plotFeatureImportance(pno2, namesno2, title = 'PredictZones : zone {0}, pollutant {1}'.format(z, 'NO2'))
        if(len(xpm10Test) > 0):
            if(not ppm10):
                print("Error : no training data for pollutant PM10 in zone {0}".format(z))
            ypm10Test = ppm10.predict(xpm10Test);
            yTest[idpm10] = ypm10Test[:];
            print("PM10 : mse : {0}".format(score_function(ypm10Test, ypm10T)))
            if(plotFeaturesInfluence):
                plotFeatureImportance(ppm10, namespm10, title = 'PredictZones : zone {0}, pollutant {1}'.format(z, 'PM10'))
        if(len(xpm2_5Test) > 0):
            if(not pm2_5l):
                print("Error : no training data for pollutant PM2_5 in zone {0}".format(z))
            ypm2_5Test = ppm2_5.predict(xpm2_5Test);
            yTest[idpm2_5] = ypm2_5Test[:];
            print("PM2_5 : mse : {0}".format(score_function(ypm2_5Test, ypm2_5T)))
            if(plotFeaturesInfluence):
                plotFeatureImportance(ppm2_5, namespm2_5, title = 'PredictZones : zone {0}, pollutant {1}'.format(z, 'PM2_5'))

        # print('Zone {2} : Check length : {0} == {1} ?'.format(len(dataTestZone.index), yno2Test.size + ypm2_5Test.size + ypm10Test.size, z))

    return arrayToResult(yTest, dataTest);

def predictGlobal(dataTrain, dataTest):



    xno2Train, yno2Train = getLearningPollutantData(getPollutant(dataTrain, 'NO2'), 'NO2');
    xpm10Train, ypm10Train = getLearningData(getPollutant(dataTrain, 'PM10'), '');
    xpm2_5Train, ypm2_5Train = getLearningData(getPollutant(dataTrain, 'PM2_5'), statiques = False);

    #Training of the gradient boosting
    pno2 = ensemble.GradientBoostingRegressor(**params)
    ppm10 = ensemble.GradientBoostingRegressor(**params)
    ppm2_5 = ensemble.GradientBoostingRegressor(**params)

    pno2.fit(xno2Train, yno2Train);
    ppm10.fit(xpm10Train, ypm10Train);
    ppm2_5.fit(xpm2_5Train, ypm2_5Train);


    #Testing part
    xno2Test, yno2Test = getLearningData(getPollutant(dataTest, 'NO2'), statiques = False);
    xpm10Test, ypm10Test = getLearningData(getPollutant(dataTest, 'PM10'), statiques = False);
    xpm2_5Test, ypm2_5Test = getLearningData(getPollutant(dataTest, 'PM2_5'), statiques = False);
    idno2 = getIdPollutant(dataTest, 'NO2');
    idpm10 = getIdPollutant(dataTest, 'PM10');
    idpm2_5 = getIdPollutant(dataTest, 'PM2_5');

    if(len(xno2Test) > 0):
        yno2Test = pno2.predict(xno2Test);
    if(len(xpm10Test) > 0):
        ypm10Test = ppm10.predict(xpm10Test);
    if(len(xpm2_5Test) > 0):
        ypm2_5Test = ppm2_5.predict(xpm2_5Test);

    nTest = len(dataTest.index);
    yTest = np.zeros(nTest);

    yTest[idno2] = yno2Test[:];
    yTest[idpm10] = ypm10Test[:];
    yTest[idpm2_5] = ypm2_5Test[:];

    print('Check length : {0} == {1} ?'.format(nTest, yno2Test.size + ypm2_5Test.size + ypm10Test.size))

    return arrayToResult(yTest, dataTest);

#marche bien sauf pour la zone 2 !!!
def predictNO2(dataTrain, dataTest, result, isTest = False):

    yTest = result['TARGET'].as_matrix();

    datano2 = getPollutant(dataTrain, 'NO2')
    datano2Train, yno2Train = getLearningPollutantData(datano2, 'NO2', removeThings = True);
    xno2Train = datano2Train.as_matrix();
    names = datano2Train.columns;

    print("Predicting NO2 : {0} data".format(len(xno2Train)))

    #Training of the gradient boosting
    pno2 = ensemble.GradientBoostingRegressor(**params)
    pno2.fit(xno2Train, yno2Train);

    #Testing part
    xno2Test, yno2T = getLearningPollutantData(getPollutant(dataTest, 'NO2'), 'NO2', removeThings = True);

    idno2 = getIdPollutant(dataTest, 'NO2');

    if(len(xno2Test) > 0):
        yno2Test = pno2.predict(xno2Test);

    if(isTest):
        print("mse : {0}".format(score_function(yno2Test, yno2T)))

        plotDeviance(pno2, xno2Test, yno2T, yno2Test);
        plotFeatureImportance(pno2, names);

    yTest[idno2] = yno2Test[:];

    return arrayToResult(yTest, dataTest);

def predictNO2Zone2(dataTrain, dataTest, result, isTest = False):
    yTest = result['TARGET'].as_matrix();

    datano2 = getZone(dataTrain, 2);
    datano2 = getPollutant(datano2, 'NO2')
    datano2Train, yno2Train = getLearningPollutantData(datano2, 'NO2');
    xno2Train = datano2Train.as_matrix();
    names = datano2Train.columns;

    print("Predicting NO2 Zone 2 : {0} data".format(len(xno2Train)))

    #Training of the gradient boosting
    pno2 = ensemble.GradientBoostingRegressor(**params)
    pno2.fit(xno2Train, yno2Train);

    #Testing part
    datano2 = getZone(dataTest, 2);
    xno2Test, yno2T = getLearningPollutantData(getPollutant(datano2, 'NO2'), 'NO2');
    xno2Test = xno2Test.as_matrix();

    idno2 = getIdPollutant(datano2, 'NO2');

    if(len(xno2Test) > 0):
        yno2Test = pno2.predict(xno2Test);

    print("mse : {0}".format(score_function(yno2Test, yno2T)))

    if(isTest):
        plotDeviance(pno2, xno2Test, yno2T, yno2Test);
        plotFeatureImportance(pno2, names);

    yTest[idno2] = yno2Test[:];

    return arrayToResult(yTest, dataTest);

def predictPollutant(dataTrain, dataTest,  plotFeaturesInfluence = False, plotDeviance = False, isTest = False):

    #initialisation of the results
    nTest = len(dataTest.index);
    yTest = np.zeros(nTest);

    for p in pollutants:

        #Lecture de l'entree
        datap = getPollutant(dataTrain, p)
        datapTrain, ypTrain = getLearningPollutantData(datap, p);
        xpTrain = datapTrain;
        names = datapTrain.columns;

        print("Predicting {1} : {0} data".format(len(xpTrain), p))

        #Training of the gradient boosting
        pp = ensemble.GradientBoostingRegressor(**params)
        pp.fit(xpTrain, ypTrain);

        #Testing part
        dataptest = getPollutant(dataTest, p)
        dpTest, ypT = getLearningPollutantData(getPollutant(dataptest, p), p);
        xpTest = dpTest;
        names = dpTest.columns

        idp = getIdPollutant(dataTest, p);

        if(len(xpTest) > 0):
            ypTest = pp.predict(xpTest);
        print(ypTest);
        #Control affichage
        if(isTest):
            print("mse : {0}".format(score_function(ypTest, ypT)))

            if(plotDeviance):
                plotDeviance(pp, xpTest, ypT, ypTest, title = 'PredictPollutant : pollutant {0}'.format(p));
            if(plotFeaturesInfluence):
                plotFeatureImportance(pp, names, title = 'PredictPollutant : pollutant {0}'.format(p))

        yTest[idp] = ypTest[:];

    return arrayToResult(yTest, dataTest);

def plotDeviance(learner, xTest, yTest, yPredicted, title = '', showDirect = False):
    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, yPredicted in enumerate(learner.staged_predict(xTest)):
        test_score[i] = learner.loss_(yPredicted, yTest)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance \n'+title)
    plt.plot(np.arange(params['n_estimators']) + 1, learner.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    if(showDirect):
        plt.show();

def plotFeatureImportance(learner, names, title = '', showDirect = True):
    feature_importance = learner.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance \n'+title)
    if(showDirect):
        plt.show();

#useless
def tcheat(dataTrain, dataTest, yTest):
    dtrain = dataTrain[dataTrain.station_id == 1].copy();
    dtest = dataTest[dataTest.station_id == 1].copy();

    if('y' in dtest):
        dtest.drop('y', axis = 1, inplace = True)

    dtest['index'] = dtest.index
    dtest.set_index('daytime', inplace = True)
    dtrain.set_index('daytime', inplace = True)

    print(dtest.loc[72])

    d = dtest.merge(dtrain, left_index = True, left_on = 'pollutant', right_on = 'pollutant', right_index = True)
    d = d[['index', 'y']]

    f = dtrain[['pollutant', 'y']]
    print(d.loc[72], f[f.index == 72])

    y = yTest.copy();
    yTest[d['index'].as_matrix()] = d['y'].as_matrix()
    return yTest;

params = {'n_estimators': 200, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

TestAlgo = True;  #TestAlgo = False;

if(not TestAlgo):
    dataTrain = loadTrainData(); dataTest = loadTestData()
    result = predictPollutant(dataTrain, dataTest);
    #result = predictNO2(dataTrain, dataTest, result);
    saveResult(result, 'result.csv')
else:
    data = loadTrainData();
    recenter(data); normalise(data);
    #data = getZone(data, 2);
    stationTest = [1, 4, 5, 6, 16, 26];
    #stationTest = [20, 22, 23, 25, 28, 11]

    dataTest = data[data['station_id'].isin(stationTest)];
    dataTest.reset_index(drop = True, inplace = True)
    dataTrain = data[~data['station_id'].isin(stationTest)];
    y = dataTest['y'].as_matrix()
    nTest = len(dataTest.index)
    result = arrayToResult(np.zeros(nTest), dataTest)

    #dataTest = loadTrainData(); dataTrain = loadTrainData();

    result = predictPollutant(dataTrain, dataTest, plotFeaturesInfluence = False, isTest = True);
    getStatLearningTest(result, dataTest)
    result = predictNO2Zone2(dataTrain, dataTest, result, isTest = True);
    getStatLearningTest(result, dataTest)
    plt.show();


    saveResult(result, 'GradientBoostingTest.csv')
