from readData import *
from public_mean_square_error import *


import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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

def predictPollutant(dataTrainInit, dataTestInit,  plotFeaturesInfluence = False, plotDeviance = False, isTest = False):

    dataTrain = addTemporalValues(dataTrainInit);
    dataTest = addTemporalValues(dataTestInit);

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

def plotFeatureImportance(learner, names, title = '', showDirect = False):
    plt.figure();
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


def predictWithLinearRegression(dataTrain, dataTest):
    xno2Train, yno2Train = getLearningData(getPollutant(dataTrain, 'NO2'), statiques = False);
    xpm10Train, ypm10Train = getLearningData(getPollutant(dataTrain, 'PM10'), statiques = False);
    xpm2_5Train, ypm2_5Train = getLearningData(getPollutant(dataTrain, 'PM2_5'), statiques = False);

    pno2 = LinearRegression()
    ppm10 = LinearRegression()
    ppm2_5 = LinearRegression()

    pno2.fit(xno2Train, yno2Train);
    ppm10.fit(xpm10Train, ypm10Train);
    ppm2_5.fit(xpm2_5Train, ypm2_5Train);


    xno2Test, yno2Test = getLearningData(getPollutant(dataTest, 'NO2'), statiques = False);
    xpm10Test, ypm10Test = getLearningData(getPollutant(dataTest, 'PM10'), statiques = False);
    xpm2_5Test, ypm2_5Test = getLearningData(getPollutant(dataTest, 'PM2_5'), statiques = False);

    if yno2Test is not None and ypm10Test is not None and ypm2_5Test is not None :
        yTestTrue = pd.concat([yno2Test, ypm10Test, ypm2_5Test], axis=0).sort_index()
    else:
        yTestTrue = None

    yno2TestPredict = pd.DataFrame(pno2.predict(xno2Test));
    yno2TestPredict.index = xno2Test.index;
    ypm10TestPredict = pd.DataFrame(ppm10.predict(xpm10Test));
    ypm10TestPredict.index = xpm10Test.index
    ypm2_5TestPredict = pd.DataFrame(ppm2_5.predict(xpm2_5Test));
    ypm2_5TestPredict.index = xpm2_5Test.index;

    yTestPredict = pd.concat([yno2TestPredict, ypm10TestPredict, ypm2_5TestPredict], axis=0)

    if yTestTrue is None:
        return yTestPredict
    else:
        return yTestPredict, yTestTrue


######### separation ###############


idInfoNames = ['ID', 'zone_id', 'station_id', 'pollutant']
gFeatureNames = ['hlres_50', 'green_5000', 'hldres_50', 'route_100', 'hlres_1000',
   'route_1000', 'roadinvdist', 'port_5000', 'hldres_100', 'natural_5000',
   'hlres_300', 'hldres_300', 'route_300', 'route_500', 'hlres_500', 'hlres_100',
   'industry_1000',  'hldres_500', 'hldres_1000', 'zone_id', 'hour', 'day', 'is_calmday']#, 'daytime']
fFeatureNames = ['temperature', 'precipprobability', 'precipintensity',
   'windbearingcos','windbearingsin', 'windspeed','cloudcover',  'pressure', 'is_calmday', 'hour', 'day', 'month']

params = {'n_estimators': 400, 'max_depth': 8, 'min_samples_split': 3,
          'learning_rate': 0.01, 'loss': 'ls'}
paramsf = {'n_estimators': 400, 'max_depth': 8, 'min_samples_split': 3,
          'learning_rate': 0.01, 'loss': 'ls'}
paramsg = {'n_estimators': 400, 'max_depth': 8, 'min_samples_split': 3,
          'learning_rate': 0.01, 'loss': 'ls'}

hourSet = range(0,24);
daySet = range(0,7);

#Ne garde que les données statiques, robuste si des colonnes ont deja ete enlevee
def getFFeatures(data):
    l = set(fFeatureNames).intersection(data.columns);
    l = list(l)
    return data[l];

#Ne garde que les doonees dynamiquesrobuste si des colonnes ont deja ete enlevee
def getGFeatures(data):
    d = data.drop_duplicates(subset = gVarSet);
    l = set(gFeatureNames).intersection(d.columns);
    l = list(l)
    return d[l];

def addRealGtoData(stationTime, data):
    pass;

def getRows(s,h,day,calmday, data):
    return (data.station_id == s) & (data.hour == h) & (data.day == day) & (data.is_calmday == calmday);
gVarSet = ['station_id', 'hour', 'day', 'is_calmday'];

def realGtoG(realg, data):
    stations = data.drop_duplicates(subset = gVarSet)
    stations.loc[:,'g'] = realg;
    for s in station_idAll:
        for h in hourSet:
            for day in daySet:
                data.loc[getRows(s,h,day,True, data), 'g'] = stations.loc[getRows(s,h,day,True, stations), 'g'].as_matrix();
                data.loc[getRows(s,h,day,False, data), 'g'] = stations.loc[getRows(s,h,day,False, stations), 'g'].as_matrix();
    return data.g

def gtoRealG(g, data):
    data.loc[:,'g'] = g;
    for s in station_idAll:
        for h in hourSet:
            for day in daySet:
                data.loc[getRows(s,h,day,True, data), 'g'] = np.mean(data.loc[getRows(s,h,day,True, data), 'g']);
                data.loc[getRows(s,h,day,False, data), 'g'] = np.mean(data.loc[getRows(s,h,day,False, data), 'g']);
    stations = data.drop_duplicates(subset = gVarSet)
    return stations.g

def initG(data):
    for s in station_idAll:
        for h in hourSet:
            for day in daySet:
                data.loc[getRows(s,h,day,True, data), 'g'] = np.max(data.loc[getRows(s,h,day,True, data), 'y']);
                data.loc[getRows(s,h,day,False, data), 'g'] = np.max(data.loc[getRows(s,h,day,False, data), 'y']);

def plotGraphG(data, calmday):
    for s in station_idTrain:
        d = data[(data.station_id == s) & (data.is_calmday == calmday)];
        if(len(d.index) < 168):
            continue;
        d = d.drop_duplicates(subset = ['hour', 'day']);
        g = d.g.as_matrix();
        h = d.hour.as_matrix();
        d = d.day.as_matrix();
        t = np.zeros(len(g));
        for i in range(len(g)):
            t[i] = d[i]*24+h[i];

        plt.figure();
        plt.plot(t,g, 'ro');
        plt.title('station {0}'.format(s))


def separation(dataTrainInit, dataTestInit, p, result):

    dataTrain = addTemporalValues(dataTrainInit);
    dataTest = addTemporalValues(dataTestInit);

    dataTrain = getPollutant(dataTrain, p);
    dataTest  = getPollutant(dataTest, p);

    idp = getIdPollutant(dataTestInit, p);

    #timeSet = dataTrain.drop_duplicates(subset = 'daytime').daytime.as_matrix();

    #definition des gb, y = g(s,t)-f(d)
    f = ensemble.GradientBoostingRegressor(**paramsf)
    g = ensemble.GradientBoostingRegressor(**paramsg)

    #g = g(s,t)
    dataTrain.loc[:,'g'] = 0;
    #initialisation de g
    initG(dataTrain);

    print("\n\nSeparation -- Initialised")

    gtrain = dataTrain.g.as_matrix();
    dtrain = getFFeatures(dataTrain);
    fFeatureNames = dtrain.columns;

    strain = getGFeatures(dataTrain);
    gFeatureNames = strain.columns;
    ytrain = dataTrain.y.as_matrix()
    nIter = 4;
    for i in range(nIter):
        f = ensemble.GradientBoostingRegressor(**paramsf)
        g = ensemble.GradientBoostingRegressor(**paramsg)

        #predict f
        ftrain = gtrain-ytrain;
        f.fit(dtrain, ftrain);

        fPredicted = f.predict(dtrain);
        print("f error : {0}".format(score_function(fPredicted, ftrain)))

        #Predict g
        gtrain = ytrain + fPredicted;
        gRealTrain = gtoRealG(gtrain, dataTrain);
        g.fit(strain, gRealTrain);

        gPredicted = g.predict(strain);
        print("g error : {0}".format(score_function(gPredicted, gRealTrain)))
        gtrain = realGtoG(gPredicted, dataTrain);


        if(i == nIter-1):
            plotFeatureImportance(g, gFeatureNames, showDirect = False, title = 'g features, iter : {0}'.format(i));
            plotFeatureImportance(f, fFeatureNames, showDirect = False, title = 'f features, iter : {0}'.format(i));
            plotGraphG(dataTrain, True);



    #predict for dataTest :
    stest = getGFeatures(dataTest);
    dtest = getFFeatures(dataTest);

    gRealtest = g.predict(stest);
    gtest = realGtoG(gRealtest, dataTest)

    ftest = f.predict(dtest);
    print(ftest[ftest <= 0])
    ypTest = gtest - ftest;
    yTest = result.TARGET.as_matrix();
    yTest[idp] = ypTest[:];

    return arrayToResult(yTest, dataTestInit)

TestAlgo = True; #a TestAlgo = False;

if(not TestAlgo):
    dataTrain = loadTrainData(); dataTest = loadTestData()

    result = predictPollutant(dataTrain, dataTest);

    result = separation(dataTrain, dataTest, 'NO2', result);
    saveResult(result, 'separationNo2v2.csv')
    result = separation(dataTrain, dataTest, 'PM10', result);
    result = separation(dataTrain, dataTest, 'PM2_5', result);

    saveResult(result, 'separationTotalv2.csv');
    #result = predictNO2(dataTrain, dataTest, result);
else:
    data = loadTrainData();
    # recenter(data); normalise(data);
    #data = getZone(data, 2);
    stationTest = [1, 4, 5, 6, 16, 26];
    stationTest = [20, 22, 23, 25, 28, 11];

    dataTest = data[data['station_id'].isin(stationTest)];
    dataTest.reset_index(drop = True, inplace = True)
    dataTrain = data[~data['station_id'].isin(stationTest)];
    y = dataTest['y'].as_matrix()
    nTest = len(dataTest.index)
    result = arrayToResult(np.zeros(nTest), dataTest)

    #dataTest = loadTrainData(); dataTrain = loadTrainData();

    # result = predictPollutant(dataTrain, dataTest, plotFeaturesInfluence = True, isTest = True)
    # plt.show();


    result = predictPollutant(dataTrain, dataTest,  plotFeaturesInfluence = True, isTest = True);
    getStatLearningTest(result, dataTest);
    result = separation(dataTrain, dataTest, 'NO2', result);
    getStatLearningTest(result, dataTest);
    plt.show();
    result = separation(dataTrain, dataTest, 'PM10', result);
    getStatLearningTest(result, dataTest);
    result = separation(dataTrain, dataTest, 'PM2_5', result);
    getStatLearningTest(result, dataTest);
    plt.show();
