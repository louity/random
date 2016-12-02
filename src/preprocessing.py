# coding: utf8
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


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

def separateZoneDatas(data):
    zone_ids = set(data['zone_id'])
    zone_datas = {}

    for i, zone_id in enumerate(zone_ids):
        zone_data = data.loc[data['zone_id'] == zone_id].reset_index(drop=True).dropna(axis=1, how='all')
        zone_datas[zone_id] = zone_data

    return zone_datas

def separatePollutantAndStationData(data):
    pollutant_datas = separatePollutantDatas(data)

    for key in pollutant_datas:
        pollutant_datas[key] = separateStationDatas(pollutant_datas[key])

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
