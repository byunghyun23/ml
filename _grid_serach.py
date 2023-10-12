# https://byunghyun23.tistory.com/83
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

raw_iris = datasets.load_iris()

X = raw_iris.data
y = raw_iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

best_acc = 0

final_k = None

for k in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if acc > best_acc:
        best_acc = acc
        final_k = k
    print('k:', k, 'acc:', acc)

print('final_k:', final_k)
print('best_acc:', best_acc)