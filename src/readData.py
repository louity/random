# coding: utf8
import pandas as pd
# coding: utf8
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import math
import timeit
from public_mean_square_error import *


#Tres pratique
idInfoNames = ['ID', 'zone_id', 'station_id', 'pollutant']
statiqueNames = ['hlres_50', 'green_5000', 'hldres_50', 'route_100', 'hlres_1000',
   'route_1000', 'roadinvdist', 'port_5000', 'hldres_100', 'natural_5000',
   'hlres_300', 'hldres_300', 'route_300', 'route_500', 'hlres_500', 'hlres_100',
   'industry_1000',  'hldres_500', 'hldres_1000']
dynamiqueNames = ['temperature', 'precipprobability', 'precipintensity',
   'windbearingcos','windbearingsin', 'windspeed','cloudcover',  'pressure', 'daytime', 'is_calmday']
hlres = ['hlres_50', 'hlres_100', 'hlres_1000', 'hlres_300', 'hlres_500']
hldres = ['hldres_50', 'hldres_100', 'hldres_300', 'hldres_500', 'hldres_1000']
green = ['green_5000']
natural = ['natural_5000']
route = ['route_300', 'route_500', 'route_100', 'route_1000']
port = [ 'port_5000']
roadinv = [ 'roadinvdist']
industry = ['industry_1000']
pollutants = ['NO2', 'PM10', 'PM2_5']
station_idTrain = [16,17,20,1,18,22,26,28,6,9,25,4,10,23,5,8,11]
station_idTest = [21,27,1,29,15,12,14,0,3,2,13,19]
zone_id = [0, 1, 2, 3, 4, 5]


"""Code par louis """

#Ajoute des données temporelles aux données
def addTemporalValues(data):
    daytimeToHour = lambda x: x % 24
    daytimeToDay = lambda x: math.floor(x / 24)
    daytimeToWeek = lambda x: math.floor(x / (24 * 7))
    daytimeToMonth = lambda x: math.floor(x / (24 * 7 * 4))

    daytime = data['daytime']

    hour = pd.DataFrame(daytime.apply(daytimeToHour))
    hour.columns = ['hour']
    day = pd.DataFrame(daytime.apply(daytimeToDay))
    day.columns = ['day']
    week = pd.DataFrame(daytime.apply(daytimeToWeek))
    week.columns = ['week']
    month = pd.DataFrame(daytime.apply(daytimeToMonth))
    month.columns = ['month']

    return pd.concat([data.drop('daytime', axis=1), hour, day, week, month], axis=1)



""" Code par alex"""

#Load les donnes
def loadTrainData():
    # lire les données
    X_train = pd.read_csv('data/X_train.csv')
    Y_train = pd.read_csv('data/Y_train.csv')

    #Hierarchise les labels des colonnes pour pouvoir separer statique/dynamique facilement
    data = mergeXY(X_train, Y_train);
    data.sort_index(axis = 1, inplace = True);

    interpoleRouteNan(data);
    interpoleHldresNan(data);
    interpoleHlresNan(data);
    setNaNValuesToZero(data);

    return data

def loadTestData():
    X_test = pd.read_csv('data/X_test.csv')
<<<<<<< HEAD
    X_test.sort_index(axis = 1, inplace = True);

    interpoleRouteNan(X_test);
    interpoleHldresNan(X_test);
    interpoleHlresNan(X_test);
    setNaNValuesToZero(X_test);
=======
    X_test.index = X_test['ID'].values
>>>>>>> origin/master

    return X_test;

#Fusionne les donnes x et y dans une meme dataFrame
def mergeXY(x, y):
    idInfo = x[idInfoNames];
    statique = x[statiqueNames];
    dynamique = x[dynamiqueNames];

    y.drop('ID', axis = 1, inplace = True);
    y.columns  = ['y'];

    dict = {'IdInfo' : idInfo, 'statique' : statique, 'dynamique' : dynamique, 'y' : y}
    newData = pd.concat(dict.values(), axis = 1);

    return newData;

#Renvoie toute les données de la zone correspondante
def getZone(data, zone):
    return data[data.zone_id == zone];

#Renvoie toutes les donnes sur la station correspondante
def getStation(data, station):
    return data[data.station_id == station];

#get the wanted lines with this pollutant. The possible values are : 'PM10', 'NO2' et 'PM2_5'
def getPollutant(data, pollutant):
    return data.loc[data['pollutant'] == pollutant];

#Return the list of the indices of the input lines which correspond to the given pollutant.
#Used during the training to be able to reconstruct the output correctly
def getIdPollutant(data, pollutant):
    return data['ID'].loc[data['pollutant'] == pollutant].index;

#Recenter not working yet
def recenter(data):
    columnsToRecenter = ['hlres_50', 'green_5000', 'hldres_50', 'route_100', 'hlres_1000',
       'route_1000', 'roadinvdist', 'port_5000', 'hldres_100', 'natural_5000',
       'hlres_300', 'hldres_300', 'route_300', 'route_500', 'hlres_500', 'hlres_100',
       'industry_1000',  'hldres_500', 'hldres_1000', 'temperature', 'precipprobability', 'precipintensity',
        'windspeed','cloudcover',  'pressure']

    #donnes non recentrees : windbearingcos/sin, daytime, is_calmday

    for column in columnsToRecenter:
        data[column]= data[column]-np.mean(data[column]);

#Normalise les donnees not working yet
def normalise(data):
    columnsToNormalise = ['hlres_50', 'green_5000', 'hldres_50', 'route_100', 'hlres_1000',
       'route_1000', 'roadinvdist', 'port_5000', 'hldres_100', 'natural_5000',
       'hlres_300', 'hldres_300', 'route_300', 'route_500', 'hlres_500', 'hlres_100',
       'industry_1000',  'hldres_500', 'hldres_1000', 'temperature', 'precipprobability', 'precipintensity',
        'windspeed','cloudcover',  'pressure']
    for column in columnsToNormalise:
        #m = np.var(data[column])
        m = np.max(data[column]);
        data[column] = data[column]/m;

#Ne garde que les données statiques, robuste si des colonnes ont deja ete enlevee
def getStatiques(data):
    l = set(statiqueNames).intersection(data.columns);
    l = list(l)
    return data[l];

#Ne garde que les doonees dynamiquesrobuste si des colonnes ont deja ete enlevee
def getDynamiques(data):
    l = set(dynamiqueNames).intersection(data.columns);
    l = list(l)
    return data[l];

""" Get learning data fonction """

<<<<<<< HEAD
#Return d,y avec d la dataFrame voulue et y les resultats correspondant. Just apply d.as_matrix() to send to the learning algorithm.
def getLearningData(data, unusedVariables = [], statiques = True, dynamiques = True):
=======
#Return an array which can directly be sent to the learning algorithm.
def getLearningData(data, unusedVariables = [], statiques=True, dynamiques=True, return_type='matrix'):
>>>>>>> origin/master
    """ Return an array which can directly be sent to the learning algorithm. Remove the column not appropriate
    to the learning process : ID, zone_id, station_id and pollutant
    Parameters :
        unusedVariables : to choose which column you may not want to use
        statiques/dynamiques : if you want these kind of varaibles
    """

    if('y' in data):
        y = data['y'].sort_index();
    else:
        y = None;
    d = data.copy();
    if(statiques == False):
        d.drop(statiqueNames, axis = 1, inplace = True, errors = 'ignore')
    if(dynamiques == False):
        d.drop(dynamiqueNames, axis = 1, inplace = True, errors = 'ignore')

    toDrop = ['ID', 'zone_id', 'station_id', 'pollutant', 'y'];

    d.drop(unusedVariables, axis = 1, inplace = True, errors = 'ignore');
    d.drop(toDrop, axis = 1, inplace = True, errors = 'ignore');
    d.sort_index()

<<<<<<< HEAD
    return d, y;
=======
    if return_type == 'matrix':
        return d.as_matrix(), y.as_matrix();
    else:
        return d, y
>>>>>>> origin/master

#Return d,y avec d la dataFrame voulue et y les resultats correspondant. Just apply d.as_matrix() to send to the learning algorithm.
def getLearningZoneData(data, z):
    """ Return an array which can directly be sent to the learning algorithm according to the given zone. Remove the column not appropriate
    to the learning process : ID, zone_id, station_id and pollutant. Cf info data.txt pour explications
    Parameters :
        z : zone to which correspond the data
    """

    #The same function is used for test and train data
    if('y' in data):
        y = data['y'].as_matrix();
    else:
        y = None;

    d = data.copy()

    interpoleRouteNan(d);
    interpoleHldresNan(d);
    interpoleHlresNan(d);
    #tout sauf hlres port et industry
    if(z == 0):
        d.drop(hlres, inplace = True, axis = 1, errors = 'ignore');
        d.drop(port, inplace = True, axis = 1, errors = 'ignore');
        d.drop(industry, inplace = True, axis = 1, errors = 'ignore');
    #tout sauf natural et industry
    elif(z == 1):
        d.drop(natural, inplace = True, axis = 1, errors = 'ignore');
        d.drop(industry, inplace = True, axis = 1, errors = 'ignore');
    elif(z == 2):
        pass;
    elif(z == 3):
        d.drop(port, inplace = True, axis = 1, errors = 'ignore');
    elif(z == 4):
        d.drop(port, inplace = True, axis = 1, errors = 'ignore');
        d.drop(industry, inplace = True, axis = 1, errors = 'ignore');
    elif(z == 5):
        d.drop(green, inplace = True, axis = 1, errors = 'ignore');
        d.drop(natural, inplace = True, axis = 1, errors = 'ignore');
        d.drop(industry, inplace = True, axis = 1, errors = 'ignore');

    toDrop = ['ID', 'zone_id', 'station_id', 'pollutant', 'y'];
    d = d.drop(toDrop, axis = 1, errors = 'ignore');

    return d, y;

#Return d,y avec d la dataFrame voulue et y les resultats correspondant. Just apply d.as_matrix() to send to the learning algorithm.
def getLearningPollutantData(data, p, removeThings = False):
    """ Return an array which can directly be sent to the learning algorithm according to the given pollutant. Remove the column not appropriate
    to the learning process : ID, zone_id, station_id and pollutant. Cf info data.txt pour explications
    Parameters :
        p : pollutant to which correspond the data
    """

    d = data.copy()

    # interpoleRouteNan(d);
    # interpoleHldresNan(d);
    # interpoleHlresNan(d);
    # setNaNValuesToZero(d);

    if(removeThings):
        #Present dans toute les zones -> zones 0, 4, 5 (car marche mieux dans predictZones)
        if(p == 'NO2'):
            d.drop(hlres, inplace = True, axis = 1, errors = 'ignore');
            d.drop(port, inplace = True, axis = 1, errors = 'ignore');
            d.drop(industry, inplace = True, axis = 1, errors = 'ignore');
            d.drop(green, inplace = True, axis = 1, errors = 'ignore');
            d.drop(natural, inplace = True, axis = 1, errors = 'ignore');
            #d = d[d['zone_id'].isin([0,4,5])];

        #Present dans les zones :
        elif(p == 'PM10'):
            d.drop(hlres, inplace = True, axis = 1, errors = 'ignore');
            d.drop(port, inplace = True, axis = 1, errors = 'ignore');
            d.drop(industry, inplace = True, axis = 1, errors = 'ignore');
            d.drop(green, inplace = True, axis = 1, errors = 'ignore');
            d.drop(natural, inplace = True, axis = 1, errors = 'ignore');

        #Present dans les zones 1 et 2 seuelement
        elif(p == 'PM2_5'):
            d.drop(natural, inplace = True, axis = 1, errors = 'ignore');
            d.drop(industry, inplace = True, axis = 1, errors = 'ignore');

    #The same function is used for test and train data
    if('y' in data):
        y = data['y'].as_matrix();
    else:
        y = None;

    toDrop = ['ID', 'zone_id', 'station_id', 'pollutant', 'y'];
    d = d.drop(toDrop, axis = 1, errors = 'ignore');

    return d, y;

#Print the mse error of the result for each couple (zone, pollutant)
def getStatLearningTest(result, dataTest):
    data = result.merge(dataTest, left_on = 'ID', right_on = 'ID');

    for z in zone_id:
        d = data[data['zone_id'] == z];
        s = " Zone {0} : pollutant ".format(z);
        for p in pollutants:
            dp = d[d['pollutant'] == p];
            y1 = dp['y'].as_matrix(); y2 = dp['TARGET'].as_matrix();
            if(len(y2) > 0):
                s = s+'{0} -> mse = {1}, '.format(p, score_function(y1, y2))
        print(s)

    y1 = data['y'].as_matrix(); y2 = data['TARGET'].as_matrix();
    print('\nGlobal MSE : {0}'.format(score_function(y1, y2)))

    """ Intrpolation """

#interpole the missing values of the route data using the other
def interpoleRouteNan(data):
    for station in station_idTest+station_idTrain:
        d = data[data['station_id'] == station].drop_duplicates(subset = 'station_id')
        d = d[route]; d.reset_index(drop = True, inplace = True)
        if(len(d.index) > 0):
            #Trouve une valeur non Nan.
            valeurNonNan = -1; case = 0;
            for r in route:
                v = d.loc[0, r];
                if(not np.isnan(v)):
                    valeurNonNan = v; case = r;
                    break;

            vBase = 0
            #definir une valeur de base
            if(case == 'route_100'):
                vBase = valeurNonNan/areaCircle(100);
            elif(case == 'route_300'):
                vBase = valeurNonNan/areaCircle(300);
            elif(case == 'route_500'):
                vBase = valeurNonNan/areaCircle(500);
            elif(case == 'route_1000'):
                vBase = valeurNonNan/areaCircle(1000);

            #Detecter les autres nan, interpoler
            if(np.isnan(d.loc[0, 'route_100'])):
                data.ix[data['station_id'] == station, 'route_100'] = vBase*areaCircle(100);
            if(np.isnan(d.loc[0, 'route_300'])):
                data.ix[data['station_id'] == station, 'route_300'] = vBase*areaCircle(300);
            if(np.isnan(d.loc[0, 'route_500'])):
                data.ix[data['station_id'] == station, 'route_500'] = vBase*areaCircle(500);
            if(np.isnan(d.loc[0, 'route_1000'])):
                data.ix[data['station_id'] == station, 'route_1000'] = vBase*areaCircle(1000);

#interpole the missing values of the hlres data using the other
def interpoleHlresNan(data):
    for station in station_idTest+station_idTrain:
        d = data[data['station_id'] == station].drop_duplicates(subset = 'station_id')
        d = d[hlres]; d.reset_index(drop = True, inplace = True)
        if(len(d.index) > 0):
            #Trouve une valeur non Nan.
            valeurNonNan = -1; case = 0;
            for r in hlres:
                v = d.loc[0, r];
                if(not np.isnan(v)):
                    valeurNonNan = v; case = r;
                    break;

            vBase = 0
            #definir une valeur de base
            if(case == 'hlres_100'):
                vBase = valeurNonNan/areaCircle(100);
            elif(case == 'hlres_300'):
                vBase = valeurNonNan/areaCircle(300);
            elif(case == 'hlres_500'):
                vBase = valeurNonNan/areaCircle(500);
            elif(case == 'hlres_1000'):
                vBase = valeurNonNan/areaCircle(1000);
            elif(case == 'hlres_50'):
                vBase = valeurNonNan/areaCircle(50);

            #Detecter les autres nan, interpoler
            if(np.isnan(d.loc[0, 'hlres_100'])):
                data.ix[data['station_id'] == station, 'hlres_100'] = vBase*areaCircle(100);
            if(np.isnan(d.loc[0, 'hlres_300'])):
                data.ix[data['station_id'] == station, 'hlres_300'] = vBase*areaCircle(300);
            if(np.isnan(d.loc[0, 'hlres_500'])):
                data.ix[data['station_id'] == station, 'hlres_500'] = vBase*areaCircle(500);
            if(np.isnan(d.loc[0, 'hlres_1000'])):
                data.ix[data['station_id'] == station, 'hlres_1000'] = vBase*areaCircle(1000);
            if(np.isnan(d.loc[0, 'hlres_50'])):
                data.ix[data['station_id'] == station, 'hlres_50'] = vBase*areaCircle(50);

#interpole the missing values of the hldres data using the other
def interpoleHldresNan(data):
    for station in station_idTest+station_idTrain:
        d = data[data['station_id'] == station].drop_duplicates(subset = 'station_id')
        d = d[hldres]; d.reset_index(drop = True, inplace = True)
        if(len(d.index) > 0):
            #Trouve une valeur non Nan.
            valeurNonNan = -1; case = 0;
            for r in hldres:
                v = d.loc[0, r];
                if(not np.isnan(v)):
                    valeurNonNan = v; case = r;
                    break;

            vBase = 0
            #definir une valeur de base
            if(case == 'hldres_100'):
                vBase = valeurNonNan/areaCircle(100);
            elif(case == 'hldres_300'):
                vBase = valeurNonNan/areaCircle(300);
            elif(case == 'hldres_500'):
                vBase = valeurNonNan/areaCircle(500);
            elif(case == 'hldres_1000'):
                vBase = valeurNonNan/areaCircle(1000);
            elif(case == 'hldres_50'):
                vBase = valeurNonNan/areaCircle(50);

            #Detecter les autres nan, interpoler
            if(np.isnan(d.loc[0, 'hldres_100'])):
                data.ix[data['station_id'] == station, 'hldres_100'] = vBase*areaCircle(100);
            if(np.isnan(d.loc[0, 'hldres_300'])):
                data.ix[data['station_id'] == station, 'hldres_300'] = vBase*areaCircle(300);
            if(np.isnan(d.loc[0, 'hldres_500'])):
                data.ix[data['station_id'] == station, 'hldres_500'] = vBase*areaCircle(500);
            if(np.isnan(d.loc[0, 'hldres_1000'])):
                data.ix[data['station_id'] == station, 'hldres_1000'] = vBase*areaCircle(1000);
            if(np.isnan(d.loc[0, 'hldres_50'])):
                data.ix[data['station_id'] == station, 'hldres_50'] = vBase*areaCircle(50);

#...
def setNaNValuesToZero(data):
    data.fillna(0, inplace = True);

#d = diameter
def areaCircle(d):
    return (d/2)**2*math.pi;

#return the correct format result given the array of the prediction
def arrayToResult(y, testData):
    ids = pd.Series(testData['ID'].as_matrix())
    ys = pd.Series(y)

    return pd.concat([ids, ys], keys = ['ID', 'TARGET'], axis = 1)

#print different time values not very helpful
def printTimeInfluence(data, stations = station_idTest+station_idTrain):
    print('station_id min max number')
    for i in stations:
        d = data[data['station_id'] == i].drop_duplicates(subset = 'daytime')
        d = d['daytime'];
        dt = d.as_matrix();
        if(dt.size > 0):
            print(i, min(dt), max(dt), len(dt));

#used to make info data.txt, station = -1 print all the information
def printStationParameters(data, station):
    if(station == -1):
        d = data.drop_duplicates(subset = 'station_id')
    else:
        d = data[data['station_id'] == station].drop_duplicates(subset = 'station_id')
    print(getStatiques(d))

#save the result
def saveResult(result, name = "Y_test.csv"):
    result.to_csv(name, index = False);

#test de la lecture si ce fichier est executé en tant que main
if __name__ == "__main__":



    data = loadTrainData();
    recenter(data);
    normalise(data);
    printStationParameters(data, -1)

    #print(data)


    1/0


    for z in zone_id:
        d = getZone(data,z);
        d, y = getLearningZoneData(d, z);

        d.drop_duplicates(subset = 'zone_id', inplace = True)
        print(d.columns)

    # X.drop('ID', axis = 1, inplace = True, level = 1)
    #X = getStation(X, 16)
    # print(getLearningData(X, statiques = False))

    # Xstation1 = getStation(X, 1);
    #print(Xstation1);


    # Xstation1 = recenter(Xstation1);
