from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 데이터 불러오기
raw_wine = datasets.load_wine()

# 피쳐, 타겟 데이터 지정
X = raw_wine.data
y = raw_wine.target

# 트레이닝/테스트 데이터 분할
X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)

# 데이터 표준화
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)


# 랜덤포레스트 학습
clf_rf = RandomForestClassifier(max_depth=2,
                                random_state=0)
clf_rf.fit(X_tn_std, y_tn)

# 예측
pred_rf = clf_rf.predict(X_te_std)
print(pred_rf)

# 정확도
accuracy = accuracy_score(y_te, pred_rf)
print(accuracy)

# confusion matrix 확인
conf_matrix = confusion_matrix(y_te, pred_rf)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_rf)
print(class_report)


# 배깅 학습
clf_bagging = BaggingClassifier(base_estimator=GaussianNB(),
                        n_estimators=10,
                        random_state=0)
clf_bagging.fit(X_tn_std, y_tn)

# 예측
pred_bagging = clf_bagging.predict(X_te_std)
print(pred_bagging)

# 정확도
accuracy = accuracy_score(y_te, pred_bagging)
print(accuracy)

# confusion matrix 확인
conf_matrix = confusion_matrix(y_te, pred_bagging)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_bagging)
print(class_report)