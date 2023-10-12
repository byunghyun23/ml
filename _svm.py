# https://byunghyun23.tistory.com/101
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 데이터 불러오기
raw_wine = datasets.load_wine()

# 피쳐, 타겟 데이터 지정
X = raw_wine.data
y = raw_wine.target

# 트레이닝/테스트 데이터 분할
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 표준화
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# 서포트벡터머신 학습
clf_svm_lr = svm.SVC(kernel='linear', random_state=0)
clf_svm_lr.fit(X_tn_std, y_tn)

# 예측
pred_svm = clf_svm_lr.predict(X_te_std)
print(pred_svm)

# 정확도
accuracy = accuracy_score(y_te, pred_svm)
print(accuracy)

# confusion matrix 확인
conf_matrix = confusion_matrix(y_te, pred_svm)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_svm)
print(class_report)