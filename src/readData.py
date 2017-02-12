# coding: utf8
import pandas as pd
# coding: utf8
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import math
import timeit

idInfoNames = ['ID', 'zone_id', 'station_id', 'pollutant']
statiqueNames = ['hlres_50', 'green_5000', 'hldres_50', 'route_100', 'hlres_1000',
   'route_1000', 'roadinvdist', 'port_5000', 'hldres_100', 'natural_5000',
   'hlres_300', 'hldres_300', 'route_300', 'route_500', 'hlres_500', 'hlres_100',
   'industry_1000',  'hldres_500', 'hldres_1000']
dynamiqueNames = ['temperature', 'precipprobability', 'precipintensity',
   'windbearingcos','windbearingsin', 'windspeed','cloudcover',  'pressure', 'daytime', 'is_calmday']
hlres = ['hlres_50', 'hlres_100', 'hlres_1000', 'hlres_300', 'hlres_500', 'station_id']
hldres = ['hldres_50', 'hldres_100', 'hldres_300', 'hldres_500', 'hldres_1000', 'station_id']
green = ['green_5000', 'station_id']
natural = ['natural_5000', 'station_id']
route = ['route_300', 'route_500', 'route_100', 'route_1000', 'station_id']
port = [ 'port_5000', 'station_id']
roadinv = [ 'roadinvdist', 'station_id']
industry = ['industry_1000', 'station_id']
station_idTrain = [16,17,20,1,18,22,26,28,6,9,25,4,10,23,5,8,11]
zone_id = [0, 1, 2, 3, 4, 5]


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


#Load les donnes
def loadTrainData():
    # lire les données
    X_train = pd.read_csv('data/X_train.csv')
    Y_train = pd.read_csv('data/Y_train.csv')

    #Hierarchise les labels des colonnes pour pouvoir separer statique/dynamique facilement
    data = mergeXY(X_train, Y_train);

    return data

def loadTestData():
    X_test = pd.read_csv('data/X_test.csv')
    X_test.index = X_test['ID'].values

    return X_test;

#Fusionne les donnes x et y
def mergeXY(x, y):
    idInfo = x[idInfoNames];
    statique = x[statiqueNames];
    dynamique = x[dynamiqueNames];

    y.drop('ID', axis = 1, inplace = True);
    y.columns  = ['y'];

    dict = {'IdInfo' : idInfo, 'statique' : statique, 'dynamique' : dynamique, 'y' : y}
    newData = pd.concat(dict.values(), axis = 1);

    return newData;

#Renvoie toute les données sur la station
def getCity(data, city):
    return data[data.zone_id == city];

def getStation(data, station):
    return data[data.station_id == station];

#get the wanted lines with this pollutant. The possible values are : 'PM10', 'NO2' et 'PM2_5'
def getPollutant(data, pollutant):
    return data.loc[data['pollutant'] == pollutant];

#Return the list of the indices of the input lines which correspond to the given pollutant
def getIdPollutant(data, pollutant):
    return data['ID'].loc[data['pollutant'] == pollutant].index;

#Recenter
def recenter(data):
    for column in data.columns:
        if(column != 'pollutant'):
            data[column]= data[column]-np.mean(data[column]);

#Normalise les donnees
def normalise(data):
    print('1')
    for column in data.columns:
        print('2')
        m = np.max(data[column]) - np.min(data[column]);
        data[column] = data[column].apply(lambda x: x/m);

#Ne garde que les données statistiques, suppose que la df a soit 2 level d'index, soit 1.
def getStatiques(data):
    return data[statiqueNames];

#Ne garde que les doonees dynamiques
def getDynamiques(data):
    return data[dynamiqueNames];

#Return an array which can directly be sent to the learning algorithm.
def getLearningData(data, unusedVariables = [], statiques=True, dynamiques=True, return_type='matrix'):
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

    if return_type == 'matrix':
        return d.as_matrix(), y.as_matrix();
    else:
        return d, y

def arrayToResult(y, testData):
    ids = pd.Series(testData['ID'].as_matrix())
    ys = pd.Series(y)

    return pd.concat([ids, ys], keys = ['ID', 'TARGET'], axis = 1)

#test de la lecture si ce fichier est executé en tant que main
if __name__ == "__main__":


    dataTrain = loadTrainData();

    for i in station_idTrain:
        data = dataTrain[dataTrain['station_id'] == i]
        data.drop_duplicates(subset = 'daytime', inplace = True)
        data = data['daytime'];
        dt = data.as_matrix();
        print(min(dt), max(dt), len(dt));


    # X.drop('ID', axis = 1, inplace = True, level = 1)
    #X = getStation(X, 16)
    # print(getLearningData(X, statiques = False))

    # Xstation1 = getStation(X, 1);
    #print(Xstation1);


    # Xstation1 = recenter(Xstation1);
