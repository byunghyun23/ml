# https://byunghyun23.tistory.com/82
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

raw_boston = datasets.load_boston()

X = raw_boston.data
y = raw_boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

rfg = RandomForestRegressor()
rfg.fit(X_train, y_train)

y_pred = rfg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('MAE', mae)

# Pipeline
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

raw_boston = datasets.load_boston()

X = raw_boston.data
y = raw_boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

rfg_pipline = Pipeline([
    ('scaler', StandardScaler()),
    ('RFG', RandomForestRegressor())
])

rfg_pipline.fit(X_train, y_train)

y_pred = rfg_pipline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('MAE', mae)