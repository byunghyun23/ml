# https://byunghyun23.tistory.com/100
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.metrics import f1_score
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

# 의사결정나무 학습
clf_tree = tree.DecisionTreeClassifier(random_state=0)
clf_tree.fit(X_tn_std, y_tn)

# 예측
pred_tree = clf_tree.predict(X_te_std)
print(pred_tree)

# f1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_te, pred_tree, average='macro')
print(f1)

# confusion matrix 확인
conf_matrix = confusion_matrix(y_te, pred_tree)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_tree)
print(class_report)