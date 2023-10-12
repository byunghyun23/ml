# https://byunghyun23.tistory.com/81
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv('train_full.csv')
X_train = train_df[train_df.columns.difference(['img_name', 'PSNR'])].to_numpy()

y_train = train_df['PSNR'].to_numpy()
y_train = y_train.reshape(len(y_train), 1)
print('X_train.shape, y_train.shape', X_train.shape, y_train.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, shuffle=False, test_size=0.2)

xg = xgb.XGBRegressor()

kfold = KFold(n_splits=5)
for train_idx, val_idx in kfold.split(X_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    xg.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric='mae', early_stopping_rounds=100)

y_pred = xg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('MAE:', mae)