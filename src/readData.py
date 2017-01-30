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

#Separe les donnees par valeur, pour pouvoir regarder
def separateDataByValues(data, column_label):
    columns_values = set(data[column_label])
    separated_datas = {}

    for i, value in enumerate(columns_values):
        value_data = data.loc[data[column_label] == value].reset_index(drop=True).drop(column_label, axis=1)
        separated_datas[value] = value_data

    return separated_datas

#Regarde les polluants
def separatePollutantDatas(data, shouldFillNaN=False):
    pollutants = set(data['pollutant'])
    pollutantDatas = {}

    fill_NaN = Imputer(missing_values=np.nan)

    for  i, pollutant in enumerate(pollutants):
        pollutant_data = data.loc[data['pollutant'] == pollutant].reset_index(drop=True).drop('pollutant', axis=1).dropna(axis=1, how='all')

        if shouldFillNaN:
            pollutant_data_fill = pd.DataFrame(fill_NaN.fit_transform(pollutant_data))
            pollutant_data_fill.columns = pollutant_data.columns
            pollutant_data = pollutant_data_fill

        pollutantDatas[pollutant] = pollutant_data

    return pollutantDatas

#Separe les stations
def separateStationDatas(data, shouldFillNaN=False):
    station_ids = set(data['station_id'])
    station_datas = {}

    fill_NaN = Imputer(missing_values=np.nan)

    for i, station_id in enumerate(station_ids):
        station_data = data.loc[data['station_id'] == station_id].reset_index(drop=True).dropna(axis=1, how='all')

        if shouldFillNaN:
            station_data_without_pollutant = station_data.drop('pollutant', axis=1)
            station_data_fill = pd.DataFrame(fill_NaN.fit_transform(station_data_without_pollutant))
            station_data_fill.columns = station_data_without_pollutant.columns
            station_data_fill = pd.concat([station_data_fill, station_data['pollutant']], axis=1);
            station_data = station_data_fill
        #station_values = station_values.sort_values('daytime', ascending=True, axis=1)

        station_datas[station_id] = station_data

    return station_datas

#Separe les zones
def separateZoneDatas(data):
    zone_ids = set(data['zone_id'])
    zone_datas = {}

    for i, zone_id in enumerate(zone_ids):
        zone_data = data.loc[data['zone_id'] == zone_id].reset_index(drop=True).dropna(axis=1, how='all')
        zone_datas[zone_id] = zone_data

    return zone_datas

def separatePollutantZoneAndDaytimeDatas(data, shouldFillNaN=False):
    pollutant_zone_datas = separatePollutantAndZoneDatas(data, shouldFillNaN)

    for pollutant_key in pollutant_zone_datas:
        for zone_key in pollutant_zone_datas[pollutant_key]:
            pollutant_zone_datas[pollutant_key][zone_key] = separateDataByValues(pollutant_zone_datas[pollutant_key][zone_key], 'daytime')
    return pollutant_zone_datas;

def separatePollutantAndZoneDatas(data, shouldFillNaN=False):
    pollutant_datas = separatePollutantDatas(data, shouldFillNaN)

    for key in pollutant_datas:
        pollutant_datas[key] = separateZoneDatas(pollutant_datas[key])

    return pollutant_datas;

def separateZoneAndStationDatas(data):
    zone_datas = separateZoneDatas(data)

    for zone_id_key in zone_datas:
        zone_datas[zone_id_key] = separateStationDatas(zone_datas[zone_id_key])

    return zone_datas

def separateZoneStationAndPollutantDatas(data):
    zone_station_datas = separateZoneAndStationDatas(data)

    for zone_id_key in zone_station_datas:
        for station_id_key in zone_station_datas[zone_id_key]:
            zone_station_datas[zone_id_key][station_id_key] = separatePollutantDatas(zone_station_datas[zone_id_key][station_id_key])

    return zone_station_datas

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

#get the wanted lines with this pollutant. The possible values are : 'PM10', 'NO2'
def getPollutant(data, pollutant):
    return data.loc[data['pollutant'] == pollutant];

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
def getLearningData(data, unusedVariables = [], statiques = True, dynamiques = True):
    """ Return an array which can directly be sent to the learning algorithm. Remove the column not appropriate
    to the learning process : ID, zone_id, station_id and pollutant
    Parameters :
        unusedVariables : to choose which column you may not want to use
        statiques/dynamiques : if you want these kind of varaibles
    """

    y = data['y'];
    d = data.copy();
    if(statiques == False):
        d.drop(statiqueNames, axis = 1, inplace = True, errors = 'ignore')
    if(dynamiques == False):
        d.drop(dynamiqueNames, axis = 1, inplace = True, errors = 'ignore')

    toDrop = ['ID', 'zone_id', 'station_id', 'pollutant', 'y'];

    d.drop(unusedVariables, axis = 1, inplace = True, errors = 'ignore');
    d.drop(toDrop, axis = 1, inplace = True, errors = 'ignore');

    return d.as_matrix(), y.as_matrix();

#test de la lecture si ce fichier est executé en tant que main
if __name__ == "__main__":

    df = pd.DataFrame(np.random.randn(3,3), index = ['a', 'b', 'c'], columns = ['c1', 'c2', 'c3']);
    print(df);
    print(df[['c1']]);
    tuples = [('1', 'c1'), ('1', 'c3'), ('2', 'c2')];
    index = pd.MultiIndex.from_tuples(tuples);
    print(df.as_matrix())

    data = loadTrainData();
    d = getPollutant(data, 'NO2')
    print(d['pollutant'])
    #print(getStatiques(data));
    #print(getDynamiques(data));
    x, y = getLearningData(data)
    print(x.shape)
    print(y.shape)

    testData = loadTestData();
    print(testData.shape)

    # X.drop('ID', axis = 1, inplace = True, level = 1)
    #X = getStation(X, 16)
    # print(getLearningData(X, statiques = False))

    # Xstation1 = getStation(X, 1);
    #print(Xstation1);


    # Xstation1 = recenter(Xstation1);
