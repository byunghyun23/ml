# https://byunghyun23.tistory.com/102
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 데이터 불러오기
raw_iris = datasets.load_iris()

# 피쳐, 타겟 데이터 지정
X = raw_iris.data
y = raw_iris.target

# 트레이닝/테스트 데이터 분할
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 표준화
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# 보팅 학습
clf1 = LogisticRegression(multi_class='multinomial',
                          random_state=1)
clf2 = svm.SVC(kernel='linear',
               random_state=1)
clf3 = GaussianNB()

clf_voting = VotingClassifier(
                estimators=[
                    ('lr', clf1),
                    ('svm', clf2),
                    ('gnb', clf3)
                ],
                voting='hard',
                weights=[1,1,1])
clf_voting.fit(X_tn_std, y_tn)

# 예측
pred_voting = clf_voting.predict(X_te_std)
print(pred_voting)

# 정확도
accuracy = accuracy_score(y_te, pred_voting)
print(accuracy)

# confusion matrix 확인
conf_matrix = confusion_matrix(y_te, pred_voting)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_voting)
print(class_report)