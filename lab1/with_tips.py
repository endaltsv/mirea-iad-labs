# Импорт необходимых библиотек
import numpy as np               # Для работы с массивами и линейной алгеброй
import pandas as pd              # Для работы с таблицами данных
import matplotlib.pyplot as plt  # Для построения графиков

# Шаг 1: Загрузка датасета Iris
url = "iris.data"  # Путь к файлу с данными
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']  # Названия столбцов
dataset = pd.read_csv(url, names=names)  # Чтение данных из файла и присвоение названий столбцам
print(dataset.head())  # Вывод первых пяти строк датасета для ознакомления


# Шаг 2: Стандартизация данных
from sklearn.preprocessing import StandardScaler  # Импорт стандартизатора

# Разделяем данные на признаки и метки классов
X = dataset.iloc[:, :-1].values  # Все столбцы, кроме последнего (признаки)
y = dataset.iloc[:, -1].values   # Последний столбец (метки классов)

# Создаем объект стандартизатора
scaler = StandardScaler()
scaler.fit(X)               # Вычисляем среднее и стандартное отклонение по каждому признаку
X_standardized = scaler.transform(X)  # Применяем стандартизацию к данным


# Шаг 3: Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split  # Импорт функции для разделения данных

# Разделяем данные в соотношении 80% на обучение и 20% на тест
X_train, X_test, y_train, y_test = train_test_split(
    X_standardized, y, test_size=0.20, random_state=42)


# Шаг 4: Расчет ядер классов и матриц ковариации
classes = np.unique(y_train)  # Получаем список уникальных классов

# Инициализируем словари для хранения средних значений и матриц ковариации для каждого класса
class_means = {}        # Словарь для средних значений признаков
class_covariances = {}  # Словарь для матриц ковариации

epsilon = 1e-5  # Малое число для регуляризации матрицы ковариации

# Проходим по каждому классу и вычисляем ядро и матрицу ковариации
for cls in classes:
    X_cls = X_train[y_train == cls]  # Выбираем объекты текущего класса
    class_means[cls] = np.mean(X_cls, axis=0)  # Вычисляем среднее значение признаков
    cov_matrix = np.cov(X_cls, rowvar=False)   # Вычисляем матрицу ковариации
    # Добавляем регуляризацию для избежания проблем с обратимостью матрицы
    class_covariances[cls] = cov_matrix + epsilon * np.eye(X_cls.shape[1])

# Шаг 5: Классификация объектов тестовой выборки
from scipy.spatial.distance import mahalanobis  # Импорт функции для вычисления Махаланобисова расстояния

# Инициализируем списки для хранения предсказанных меток классов
y_pred_euclidean = []  # Предсказания по Евклидовой метрике
y_pred_mahalanobis = []  # Предсказания по метрике Махаланобиса

# Проходим по каждому объекту из тестовой выборки
for x in X_test:
    # Вычисляем Евклидовы расстояния до ядер классов
    distances_euclidean = []
    for cls in classes:
        dist_euclid = np.linalg.norm(x - class_means[cls])  # Евклидово расстояние
        distances_euclidean.append(dist_euclid)
    # Определяем класс с минимальным Евклидовым расстоянием
    y_pred_euclidean.append(classes[np.argmin(distances_euclidean)])

    # Вычисляем Махаланобисовы расстояния до ядер классов
    distances_mahalanobis = []
    for cls in classes:
        diff = x - class_means[cls]  # Разность между объектом и ядром класса
        cov_inv = np.linalg.inv(class_covariances[cls])  # Обратная матрица ковариации
        dist_mahalanobis = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))  # Махаланобисово расстояние
        distances_mahalanobis.append(dist_mahalanobis)
    # Определяем класс с минимальным Махаланобисовым расстоянием
    y_pred_mahalanobis.append(classes[np.argmin(distances_mahalanobis)])


# Шаг 6: Оценка качества классификации
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Импорт метрик

# Метрики для Евклидовой метрики
print("Метрики для Евклидовой метрики:")
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)  # Точность
print(f"Точность: {accuracy_euclidean:.4f}")
print("Матрица неточностей:")
print(confusion_matrix(y_test, y_pred_euclidean))  # Матрица неточностей
print("Отчет о классификации:")
print(classification_report(y_test, y_pred_euclidean))  # Подробный отчет

# Метрики для метрики Махаланобиса
print("\nМетрики для метрики Махаланобиса:")
accuracy_mahalanobis = accuracy_score(y_test, y_pred_mahalanobis)  # Точность
print(f"Точность: {accuracy_mahalanobis:.4f}")
print("Матрица неточностей:")
print(confusion_matrix(y_test, y_pred_mahalanobis))  # Матрица неточностей
print("Отчет о классификации:")
print(classification_report(y_test, y_pred_mahalanobis))  # Подробный отчет


# Дополнительный анализ: Использование KNN-классификатора
from sklearn.neighbors import KNeighborsClassifier  # Импорт KNN-классификатора

# Создаем и обучаем KNN-классификатор с заданными параметрами
classifier = KNeighborsClassifier(n_neighbors=20, metric='euclidean', weights='distance')
classifier.fit(X_train, y_train)  # Обучаем модель на обучающей выборке

# Предсказываем метки классов для тестовой выборки
y_pred_knn = classifier.predict(X_test)

# Метрики качества для KNN-классификатора
print("\nМетрики для KNN-классификатора:")
accuracy_knn = accuracy_score(y_test, y_pred_knn)  # Точность
print(f"Точность: {accuracy_knn:.4f}")
print("Матрица неточностей:")
print(confusion_matrix(y_test, y_pred_knn))  # Матрица неточностей
print("Отчет о классификации:")
print(classification_report(y_test, y_pred_knn))  # Подробный отчет


# Исследование зависимости ошибки от количества соседей K
error = []  # Список для хранения ошибок

# Перебираем значения K от 1 до 40
for i in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=i)  # Создаем KNN-классификатор с текущим K
    knn.fit(X_train, y_train)  # Обучаем классификатор
    pred_i = knn.predict(X_test)  # Предсказываем метки для тестовой выборки
    error_rate = np.mean(pred_i != y_test)  # Вычисляем среднюю ошибку
    error.append(error_rate)  # Добавляем ошибку в список

# Построение графика зависимости ошибки от K
plt.figure(figsize=(12, 6))  # Устанавливаем размер графика
plt.plot(range(1, 41), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Зависимость ошибки от количества соседей K')  # Заголовок графика
plt.xlabel('Количество соседей K')  # Подпись оси X
plt.ylabel('Средняя ошибка')        # Подпись оси Y
plt.show()  # Отображение графика
