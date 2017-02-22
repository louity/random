from readData import *
from public_mean_square_error import *


import numpy as np
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

def predict(dataTrain, dataTest):
    xno2Train, yno2Train = getLearningData(getPollutant(dataTrain, 'NO2'), statiques = False);
    xpm10Train, ypm10Train = getLearningData(getPollutant(dataTrain, 'PM10'), statiques = False);
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

def predictWithLinearRegression(dataTrain, dataTest, apply_log=False):
    xno2Train, yno2Train = getLearningData(getPollutant(dataTrain, 'NO2'), statiques = False, return_type='dataframe');
    xpm10Train, ypm10Train = getLearningData(getPollutant(dataTrain, 'PM10'), statiques = False, return_type='dataframe');
    xpm2_5Train, ypm2_5Train = getLearningData(getPollutant(dataTrain, 'PM2_5'), statiques = False, return_type='dataframe');

    if apply_log:
        yno2Train = yno2Train.apply(np.log)
        ypm10Train = ypm10Train.apply(np.log);
        ypm2_5Train = ypm2_5Train.apply(np.log);

    #Training of the gradient boosting
    pno2 = LinearRegression()
    ppm10 = LinearRegression()
    ppm2_5 = LinearRegression()

    pno2.fit(xno2Train, yno2Train);
    ppm10.fit(xpm10Train, ypm10Train);
    ppm2_5.fit(xpm2_5Train, ypm2_5Train);


    #Testing part
    xno2Test, yno2Test = getLearningData(getPollutant(dataTest, 'NO2'), statiques = False, return_type='dataframe');
    xpm10Test, ypm10Test = getLearningData(getPollutant(dataTest, 'PM10'), statiques = False, return_type='dataframe');
    xpm2_5Test, ypm2_5Test = getLearningData(getPollutant(dataTest, 'PM2_5'), statiques = False, return_type='dataframe');

    if yno2Test is not None and ypm10Test is not None and ypm2_5Test is not None :
        yTestTrue = pd.concat([yno2Test, ypm10Test, ypm2_5Test], axis=0).sort_index()
    else:
        yTestTrue = None

    yno2TestPredict = pd.DataFrame(np.exp(pno2.predict(xno2Test)));
    yno2TestPredict.index = xno2Test.index;
    ypm10TestPredict = pd.DataFrame(np.exp(ppm10.predict(xpm10Test)));
    ypm10TestPredict.index = xpm10Test.index
    ypm2_5TestPredict = pd.DataFrame(np.exp(ppm2_5.predict(xpm2_5Test)));
    ypm2_5TestPredict.index = xpm2_5Test.index;

    yTestPredict = pd.concat([yno2TestPredict, ypm10TestPredict, ypm2_5TestPredict], axis=0)

    return yTestPredict, yTestTrue

def predictWithMeanValue(dataTrain, dataTest, apply_log=False):
    means = {
        'NO2': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0},
        'PM10': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0},
        'PM2_5': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    }

    for pollutant in means.keys():
        for zone_id in means[pollutant].keys():
            values = dataTrain[dataTrain['pollutant'] == pollutant][dataTrain['zone_id'] == int(zone_id)]['y'].values
            means[pollutant][zone_id] = np.mean(values)

    index = dataTest.index
    yTest = pd.DataFrame(index=index, data=np.zeros_like(index), columns=['TARGET'])

    for i in index:
        zone_id = str(int(dataTest.get_value(i, 'zone_id')))
        pollutant = dataTest.get_value(i, 'pollutant')

        yTest.set_value(i, 'TARGET', means[pollutant][zone_id])

    return yTest, means

def predictWithMeanValue2(dataTrain, dataTest, apply_log=False):
    means = {
        'NO2': np.mean(dataTrain[dataTrain['pollutant'] == 'NO2']['y'].values),
        'PM10': np.mean(dataTrain[dataTrain['pollutant'] == 'PM10']['y'].values),
        'PM2_5': np.mean(dataTrain[dataTrain['pollutant'] == 'PM2_5']['y'].values)
    }


    index = dataTest.index
    yTest = pd.DataFrame(index=index, data=np.zeros_like(index), columns=['TARGET'])

    for i in index:
        pollutant = dataTest.get_value(i, 'pollutant')
        daytime = dataTest.get_value(i, 'daytime')

        daytime_data = dataTrain[dataTrain['pollutant'] == pollutant][dataTrain['daytime'] == daytime]['y']

        if len(daytime_data) > 0:
            yTest.set_value(i, 'TARGET', np.mean(daytime_data.values))
        else:
            yTest.set_value(i, 'TARGET', means['pollutant'])

    return yTest, means


data = loadTrainData();

stationTest = [4, 10, 22];

#dataTest = data[data['station_id'].isin(stationTest)];
#dataTrain = data[~data['station_id'].isin(stationTest)];


dataTrain = loadTrainData()
dataTest = loadTestData()

#result = predict(dataTrain, dataTest);

#yTestPredicted, yTestTrue = predictWithLinearRegression(dataTrain, dataTest, apply_log=True);
yTest, means = predictWithMeanValue2(dataTrain, dataTest)

#yPredicted = result['TARGET'].as_matrix();

#mse = score_function(yTestPredicted.values, yTestTrue.values)
#print("MSE: %.4f" % mse)
