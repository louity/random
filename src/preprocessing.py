# coding: utf8
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


def separatePollutantDatas(data):
    pollutants = set(data['pollutant'])
    pollutantDatas = {}

    fill_NaN = Imputer(missing_values=np.nan)

    for  i, pollutant in enumerate(pollutants):
        pollutant_data = data.loc[data['pollutant'] == pollutant]
        pollutant_data = pollutant_data.reset_index(drop=True).drop('pollutant', axis=1).dropna(axis=1, how='all')
        pollutant_data_fill = pd.DataFrame(fill_NaN.fit_transform(pollutant_data))
        pollutant_data_fill.columns = pollutant_data.columns

        pollutantDatas[pollutant] = pollutant_data_fill

    return pollutantDatas

def separateStationDatas(data):
    station_ids = set(data['station_id'])
    stationDatas = {}

    for i, station_id in enumerate(station_ids):
        station_values = data.loc[data['station_id'] == station_id].reset_index(drop=True)
        #station_values = station_values.sort_values('daytime', ascending=True, axis=1)

        stationDatas[station_id] = station_values

    return stationDatas
