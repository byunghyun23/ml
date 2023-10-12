# https://byunghyun23.tistory.com/90
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix, classification_report

raw_cancer = datasets.load_breast_cancer()

X = raw_cancer.data
y = raw_cancer.target
# X.shape, y.shape: (569, 30) (569,)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape: (426, 30) (143, 13) (426,) (143,)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

clf_logistic_l2 = LogisticRegression(penalty='l2')
clf_logistic_l2.fit(X_train, y_train)

pred_logistic = clf_logistic_l2.predict(X_test)
# [0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0 0 0 1 1 0 1 1 0 1 0 1 0 1 0 1 0 1
#  0 1 0 0 1 0 1 1 0 1 1 1 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 0 1 0 0 0 1 1 0 1 0
#  0 1 1 1 1 1 0 0 0 1 0 1 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1
#  0 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 0 0 0 1 1 1 0]
print(pred_logistic)

pred_prob_logistic = clf_logistic_l2.predict_proba(X_test)
# [[9.98638613e-01 1.36138656e-03]
#  [3.95544804e-02 9.60445520e-01]
#  [1.30896362e-03 9.98691036e-01]
#  [1.24473354e-02 9.87552665e-01]
#  ...
#  [8.41453252e-05 9.99915855e-01]
#  [1.58701592e-03 9.98412984e-01]
#  [1.26424968e-03 9.98735750e-01]
#  [9.99999994e-01 5.81805301e-09]]
print(pred_prob_logistic)

precision = precision_score(y_test, pred_logistic)
# 0.9666666666666667
print(precision)

conf_matrix = confusion_matrix(y_test, pred_logistic)
# [[50  3]
#  [ 3 87]]
print(conf_matrix)

class_report = classification_report(y_test, pred_logistic)
#               precision    recall  f1-score   support
#
#            0       0.94      0.94      0.94        53
#            1       0.97      0.97      0.97        90
#
#     accuracy                           0.96       143
#    macro avg       0.96      0.96      0.96       143
# weighted avg       0.96      0.96      0.96       143
print(class_report)