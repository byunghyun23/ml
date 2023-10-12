# https://byunghyun23.tistory.com/85
from sklearn.metrics import confusion_matrix
y_pred = [1, 1, 3, 2]
y_true = [0, 1, 2, 3]
print(confusion_matrix(y_true, y_pred))

# Accuracy
from sklearn.metrics import accuracy_score

y_pred = [1, 1, 3, 2]
y_true = [0, 1, 2, 3]

print(accuracy_score(y_true, y_pred, normalize=False))      # 1
print(accuracy_score(y_true, y_pred))                       # 0.25

# Classification Report
from sklearn.metrics import classification_report

y_pred = [1, 1, 3, 2]
y_true = [0, 1, 2, 3]

target_names = ['class_0', 'class_1', 'class_2', 'class_3']

print(classification_report(y_true, y_pred, target_names=target_names))


# MAE
from sklearn.metrics import mean_absolute_error

y_pred = [5, 1.5, -1, 9]
y_true = [4.3, 2, 1, 9]

print(mean_absolute_error(y_true, y_pred))          # 0.8

# MSE
from sklearn.metrics import mean_squared_error

y_pred = [5, 1.5, -1, 9]
y_true = [4.3, 2, 1, 9]

print(mean_squared_error(y_true, y_pred))          # 1.185
