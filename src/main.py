# coding: utf8
""" packages
    Le package panda definit des dataframe avec plein de fonctions pratiques
    voir http://pandas.pydata.org/
"""
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import Imputer

# scripts
import plotUtils
import nearestNeighbors
import preprocessing
import verification
import loadData

XY_train = preprocessing.addTemporalValues(loadData.XY_train)
X_test = preprocessing.addTemporalValues(loadData.X_test)
#nearestNeighbors.nearest_neighbors_by_pollutant()
#nearestNeighbors.nearest_neighbors_by_pollutant_and_zone()
nearestNeighbors.nearest_neighbors_by_pollutant(train_data=XY_train, test_data=X_test)
#zone_station_pollutant_datas = preprocessing.separateZoneStationAndPollutantDatas(loadData.XY_train)
#plotUtils.plot_zone_station_values(zone_station_pollutant_datas, 1.0, 'NO2')

