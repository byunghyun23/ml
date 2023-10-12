# https://byunghyun23.tistory.com/80
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import numpy


X = np.array([1, 2, 5, 8, 10])
y = np.array([10, 15, 68, 80, 95])

mean_X = np.mean(X)
mean_Y = np.mean(y)

dev_X = X - mean_X
dev_Y = y - mean_Y

a = sum(dev_X * dev_Y) / sum(dev_X**2)
b = np.mean(y) - (a * np.mean(X))
y_pred = a * X + b

print(y)		# [10 15 68 80 95]
print(y_pred)		# [ 12.42857143  22.23129252  51.63945578  81.04761905 100.65306122]

MAE = mean_absolute_error(y, y_pred)

print(MAE)		# 48.4



from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor()
etr.fit(X.reshape(-1, 1), y.reshape(-1, 1))
etr_y_pred = etr.predict(np.arange(1, 11, 1).reshape(-1, 1))
print(etr_y_pred)

# plt.scatter(X, y)
# plt.plot(X, y_pred, 'g-')
# plt.plot(np.arange(1, 11, 1), etr_y_pred, 'r-')
# plt.legend(['y', 'y_pred', 'etr_y_pred'])
# plt.xlabel('Study time')
# plt.ylabel('Score')
# plt.show()