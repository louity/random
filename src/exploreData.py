# coding: utf8
import pandas as pd
import readData
import matplotlib.pyplot as plt
import numpy as np

data = readData.loadTrainData();


dynamic_keys_to_plot = ['temperature', 'precipintensity', 'windspeed','cloudcover', 'pressure']

N_BINS = 30
for key in dynamic_keys_to_plot:
    plt.figure(1)
    plt.hist(data[key].values, bins=N_BINS)
    plt.title(key)
    plt.figure(2)
    plt.hist(data[key].values, log=True, bins=N_BINS)
    plt.title(key + ' on log scale')
    plt.show()


#   NO2.head(1000).plot(kind='scatter', x='y', y='daytime')
#   PM2.plot(kind='scatter', x='y', y='daytime')
#   plt.show()

#   NO2.head(2000).plot(kind='scatter', x='y', y='windspeed')
#   plt.show()

#   table = pd.crosstab(NO2['LOG_TARGET'], NO2['windspeed'])

#   NO2.boxplot(column='LOG_TARGET', by='is_calmday')
