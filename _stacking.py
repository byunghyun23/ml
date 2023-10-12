# https://byunghyun23.tistory.com/105
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 데이터 불러오기
raw_breast_cancer = datasets.load_breast_cancer()

# 피쳐, 타겟 데이터 지정
X = raw_breast_cancer.data
y = raw_breast_cancer.target

# 트레이닝/테스트 데이터 분할
X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)

# 데이터 표준화
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# 스태킹 학습
clf1 = svm.SVC(kernel='linear', random_state=1)
clf2 = GaussianNB()

clf_stkg = StackingClassifier(
            estimators=[
                ('svm', clf1),
                ('gnb', clf2)
            ],
            final_estimator=LogisticRegression())
clf_stkg.fit(X_tn_std, y_tn)

# 예측
pred_stkg = clf_stkg.predict(X_te_std)
print(pred_stkg)

# 정확도
accuracy = accuracy_score(y_te, pred_stkg)
print(accuracy)

# confusion matrix 확인
conf_matrix = confusion_matrix(y_te, pred_stkg)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_stkg)
print(class_report)