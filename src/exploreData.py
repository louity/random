# coding: utf8
import pandas as pd
import readData
import matplotlib.pyplot as plt
import numpy as np

data = readData.loadTrainData();
data = readData.getZone(data, 2);
pollutant_data = readData.separateDataByValues(data, 'pollutant')

NO2 = pollutant_data['NO2']
PM2 = pollutant_data['PM2_5']

print(NO2.columns)


# replace target by its log
LOG_TARGET = pd.DataFrame(NO2['y'].apply(np.log))
LOG_TARGET.columns = ['LOG_TARGET']
NO2 = pd.concat([NO2, LOG_TARGET], axis=1)

NO2['LOG_TARGET'].plot.hist(bins=70)
plt.title('log NO2 concentrations')
#plt.show()

NO2['LOG_TARGET'].plot(kind='box')
plt.title('log NO2 concentrations')
#plt.show()

NO2['windspeed'].plot.hist(bins=30)
plt.title('windspeed values')
#plt.show()

NO2['precipintensity'].plot.hist(bins=30)
plt.title('precipitation intensity values')
#plt.show()

NO2.head(1000).plot(kind='scatter', x='y', y='daytime')
PM2.plot(kind='scatter', x='y', y='daytime')
plt.show()

NO2.head(2000).plot(kind='scatter', x='y', y='windspeed')
plt.show()

table = pd.crosstab(NO2['LOG_TARGET'], NO2['windspeed'])

NO2.boxplot(column='LOG_TARGET', by='is_calmday')
