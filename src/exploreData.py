# coding: utf8
import pandas as pd
import loadData
import preprocessing
import matplotlib.pyplot as plt

pollutant_data = preprocessing.separateDataByValues(loadData.XY_train, 'pollutant')

NO2 = pollutant_data['NO2']

print NO2.columns

print 'NO2 mean and std', NO2.mean(), NO2.std()

# replace target by its log
LOG_TARGET = pd.DataFrame(NO2['TARGET'].apply(np.log))
LOG_TARGET.columns = ['LOG_TARGET']
NO2 = pd.concat([NO2, LOG_TARGET], axis=1)

NO2['LOG_TARGET'].plot.hist(bins=70)
plt.title('log NO2 concentrations')
plt.show()

NO2['LOG_TARGET'].plot(kind='box')
plt.title('log NO2 concentrations')
plt.show()

NO2['windspeed'].plot.hist(bins=30)
plt.title('windspeed values')
plt.show()

NO2['precipintensity'].plot.hist(bins=30)
plt.title('precipitation intensity values')
plt.show()

NO2.head(1000).plot(kind='scatter', x='TARGET', y='precipintensity')
plt.show()

NO2.head(2000).plot(kind='scatter', x='TARGET', y='windspeed')
plt.show()

table = pd.crosstab(NO2['LOG_TARGET'], NO2['windspeed'])
print table

NO2.boxplot(column='LOG_TARGET', by='is_calmday')
