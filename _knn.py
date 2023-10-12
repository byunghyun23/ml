# https://byunghyun23.tistory.com/87
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

raw_iris = datasets.load_iris()

X = raw_iris.data
y = raw_iris.target
# X.shape, y.shape: (150, 4) (150,)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=144)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape: (112, 4) (38, 4) (112,) (38,)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

clf_knn = KNeighborsClassifier(n_neighbors=2)
clf_knn.fit(X_train, y_train)

clf_knn_pred = clf_knn.predict(X_test)
# [1 2 0 2 2 0 1 2 2 0 1 1 2 0 1 0 2 1 1 2 0 0 1 1 1 2 2 1 2 1 1 0 2 1 2 0 2 1]

accuracy = accuracy_score(y_test, clf_knn_pred)
# 1.0

conf_matrix = confusion_matrix(y_test, clf_knn_pred)
# [[ 9  0  0]
#  [ 0 15  0]
#  [ 0  0 14]]

class_report = classification_report(y_test, clf_knn_pred)
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00         9
#            1       1.00      1.00      1.00        15
#            2       1.00      1.00      1.00        14
#
#     accuracy                           1.00        38
#    macro avg       1.00      1.00      1.00        38
# weighted avg       1.00      1.00      1.00        38