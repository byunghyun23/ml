# https://byunghyun23.tistory.com/88
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

raw_boston = datasets.load_boston()

X = raw_boston.data
y = raw_boston.target
# X.shape, y.shape: (506, 13) (506,)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape: (379, 13) (127, 13) (379,) (127,)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.coef_)
# [-1.07145146  1.34036243  0.26298069  0.66554537 -2.49842551  1.97524314
#   0.19516605 -3.14274974  2.66736136 -1.80685572 -2.13034748  0.56172933
#  -4.03223518]
print(lr.intercept_)
# 22.344591029023768

pred_lr = lr.predict(X_test)

mse = mean_squared_error(y_test, pred_lr)
# 21.89776539604949
print(mse)

