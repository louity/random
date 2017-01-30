from readData import *
from public_mean_square_error import *


import numpy as np
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

data = loadTrainData();
y = data['y'].as_matrix();

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


#dataTrain = loadTrainData(); dataTest = loadTestData()

data = loadTrainData();

stationTest = [4, 10, 22];

dataTest = data[data['station_id'].isin(stationTest)];
dataTest.reset_index(drop = True, inplace = True)
dataTrain = data[~data['station_id'].isin(stationTest)];
y = dataTest['y'].as_matrix()

result = predict(dataTrain, dataTest);

yPredicted = result['TARGET'].as_matrix();

mse = score_function(y, yPredicted)
print("MSE: %.4f" % mse)
