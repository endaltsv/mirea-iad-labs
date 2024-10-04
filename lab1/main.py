import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


url = "iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)
print(dataset.head())

from sklearn.preprocessing import StandardScaler

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

scaler = StandardScaler()
scaler.fit(X)
X_standardized = scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_standardized, y, test_size=0.20, random_state=42)

classes = np.unique(y_train)

class_means = {}
class_covariances = {}

epsilon = 1e-5

for cls in classes:
    X_cls = X_train[y_train == cls]
    class_means[cls] = np.mean(X_cls, axis=0)
    cov_matrix = np.cov(X_cls, rowvar=False)
    class_covariances[cls] = cov_matrix + epsilon * np.eye(X_cls.shape[1])


y_pred_euclidean = []
y_pred_mahalanobis = []

for x in X_test:
    distances_euclidean = []
    for cls in classes:
        dist_euclid = np.linalg.norm(x - class_means[cls])
        distances_euclidean.append(dist_euclid)
    y_pred_euclidean.append(classes[np.argmin(distances_euclidean)])

    distances_mahalanobis = []
    for cls in classes:
        diff = x - class_means[cls]
        cov_inv = np.linalg.inv(class_covariances[cls])
        dist_mahalanobis = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
        distances_mahalanobis.append(dist_mahalanobis)
    y_pred_mahalanobis.append(classes[np.argmin(distances_mahalanobis)])


print("Метрики для Евклидовой метрики:")
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)
print(f"Точность: {accuracy_euclidean:.4f}")
print("Матрица неточностей:")
print(confusion_matrix(y_test, y_pred_euclidean))
print("Отчет о классификации:")
print(classification_report(y_test, y_pred_euclidean))

print("\nМетрики для метрики Махаланобиса:")
accuracy_mahalanobis = accuracy_score(y_test, y_pred_mahalanobis)
print(f"Точность: {accuracy_mahalanobis:.4f}")
print("Матрица неточностей:")
print(confusion_matrix(y_test, y_pred_mahalanobis))
print("Отчет о классификации:")
print(classification_report(y_test, y_pred_mahalanobis))


classifier = KNeighborsClassifier(n_neighbors=20, metric='euclidean', weights='distance')
classifier.fit(X_train, y_train)
y_pred_knn = classifier.predict(X_test)

print("\nМетрики для KNN-классификатора:")
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Точность: {accuracy_knn:.4f}")
print("Матрица неточностей:")
print(confusion_matrix(y_test, y_pred_knn))
print("Отчет о классификации:")
print(classification_report(y_test, y_pred_knn))

error = []

for i in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate = np.mean(pred_i != y_test)
    error.append(error_rate)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 41), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Зависимость ошибки от количества соседей K')
plt.xlabel('Количество соседей K')
plt.ylabel('Средняя ошибка')
plt.show()
