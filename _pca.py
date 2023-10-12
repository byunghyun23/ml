# https://byunghyun23.tistory.com/106
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 데이터 불러오기
raw_wine = datasets.load_wine()

# 피쳐, 타겟 데이터 지정
X = raw_wine.data
y = raw_wine.target

# 트레이닝/테스트 데이터 분할
X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=1)

# 데이터 표준화
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# PCA
pca = PCA(n_components=2)
pca.fit(X_tn_std)
X_tn_pca = pca.transform(X_tn_std)
X_te_pca = pca.transform(X_te_std)

# 차원축소 확인
print(X_tn_std.shape)
print(X_tn_pca.shape)

# 공분산 행렬 확인
print(pca.get_covariance())

# 고유값 확인
print(pca.singular_values_)

# 고유벡터 확인
print(pca.components_)

# 설명된 분산
print(pca.explained_variance_)

# 설명된 분산 비율
print(pca.explained_variance_ratio_)


# 데이터프레임 생성
pca_columns = ['pca_comp1', 'pca_comp2']
X_tn_pca_df = pd.DataFrame(X_tn_pca,
                           columns=pca_columns)
X_tn_pca_df['target'] = y_tn
X_tn_pca_df.head(5)


# 라벨 미적용 PCA 데이터
plt.scatter(X_tn_pca_df['pca_comp1'],
            X_tn_pca_df['pca_comp2'],
            marker='o')
plt.xlabel('pca_component1')
plt.ylabel('pca_component2')
plt.show()

# 라벨 적용 PCA 플랏
df = X_tn_pca_df
df_0 = df[df['target']==0]
df_1 = df[df['target']==1]
df_2 = df[df['target']==2]

X_11 = df_0['pca_comp1']
X_12 = df_1['pca_comp1']
X_13 = df_2['pca_comp1']

X_21 = df_0['pca_comp2']
X_22 = df_1['pca_comp2']
X_23 = df_2['pca_comp2']

target_0 = raw_wine.target_names[0]
target_1 = raw_wine.target_names[1]
target_2 = raw_wine.target_names[2]

plt.scatter(X_11, X_21,
            marker='o',
            label=target_0)
plt.scatter(X_12, X_22,
            marker='x',
            label=target_1)
plt.scatter(X_13, X_23,
            marker='^',
            label=target_2)
plt.xlabel('pca_component1')
plt.ylabel('pca_component2')
plt.legend()
plt.show()

# 반복문을 이용해 PCA 플랏
df = X_tn_pca_df
markers=['o','x','^']

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = raw_wine.target_names[i]
    X1 = df_i['pca_comp1']
    X2 = df_i['pca_comp2']
    plt.scatter(X1, X2,
                marker=mark,
                label=target_i)

plt.xlabel('pca_component1')
plt.ylabel('pca_component2')
plt.legend()
plt.show()

# PCA 적용 전/후 예측 정확도 비교
# PCA 적용 전 예측 정확도
clf_rf = RandomForestClassifier(max_depth=2,
                                random_state=0)
clf_rf.fit(X_tn_std, y_tn)

# 예측
pred_rf = clf_rf.predict(X_te_std)

# PCA 적용 전 정확도
accuracy = accuracy_score(y_te, pred_rf)
print(accuracy)

# PCA 적용 후 학습
clf_rf_pca = RandomForestClassifier(max_depth=2,
                                    random_state=0)
clf_rf_pca.fit(X_tn_pca, y_tn)

# 예측
pred_rf_pca = clf_rf_pca.predict(X_te_pca)

# PCA 적용 후 정확도
accuracy_pca = accuracy_score(y_te, pred_rf_pca)
print(accuracy_pca)