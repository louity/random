from readData import *
import public_mean_square_error



from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

p1 = ensemble.GradientBoostingRegressor(**params)
p2 = ensemble.GradientBoostingRegressor(**params)

data = loadTrainData();

datap1 = getPollutant(data, 'NO2');
xp1, yp1 = getLearningData(data, statiques = False);

p1.fit(xp1, yp1);
mse = score_function(yp1, p1.predict(xp1))
print("MSE: %.4f" % mse)
