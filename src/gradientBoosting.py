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

xTrain, yTrain = loadTrainData();

xp1 = getPollutant(xTrain, 'NO2');
xp1 = getLearningData(xp1);

p1.fit(xTrain, yTrain)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)
