# https://byunghyun23.tistory.com/89
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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

lasso = Lasso()
ridge = Ridge()
elastic_net = ElasticNet()

lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
elastic_net.fit(X_train, y_train)

print(lasso.coef_)
# [-0.          0.         -0.          0.         -0.          1.98095526
#  -0.         -0.         -0.         -0.         -1.35346816  0.
#  -3.88203158]
print(lasso.intercept_)
# 22.344591029023764

print(ridge.coef_)
# [-1.05933451  1.31050717  0.23022789  0.66955241 -2.45607567  1.99086611
#   0.18119169 -3.09919804  2.56480813 -1.71116799 -2.12002592  0.56264409
#  -4.00942448]
print(ridge.intercept_)
# 22.344591029023768

print(elastic_net.coef_)
# [-0.41185619  0.12517736 -0.242448    0.35325324 -0.43649162  1.96540308
#  -0.02116576 -0.         -0.         -0.16013619 -1.2626002   0.32546709
#  -2.36558977]
print(elastic_net.intercept_)
# 22.34459102902376

pred_lasso = lasso.predict(X_test)
pred_ridge = ridge.predict(X_test)
pred_elastic_net = elastic_net.predict(X_test)

lasso_mse = mean_squared_error(y_test, pred_lasso)
# 32.74719740278476
print(lasso_mse)

ridge_mse = mean_squared_error(y_test, pred_ridge)
# 21.894849212618745
print(ridge_mse)

elastic_net_mse = mean_squared_error(y_test, pred_elastic_net)
# 35.196183733607924
print(elastic_net_mse)

