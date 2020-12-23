from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN


def findByNestedLoops(object_list, distance, alfa):
    outline_labels_list = []
    for item_x in object_list:
        count = 0
        for item_y in object_list:
            if item_x[0] != item_y[0] and item_x[1] != item_y[1] and euclidean(item_x, item_y) < distance:
                count = count + 1
        if count >= (alfa * len(object_list)):
            outline_labels_list.append(0)
        else:
            outline_labels_list.append(1)
    plt.scatter(x=object_list[:,0], y=object_list[:,1], c=outline_labels_list)
    plt.title('График с аномалиями')
    plt.show()


# 1 Набор: Строим выпуклые кластеры
X_blobs,y_blobs = make_blobs(centers=4, n_samples=200, random_state=0, cluster_std=0.7)

noise_len = 0.03 * len(y_blobs)
X_blobs_noise = X_blobs.copy()
for i in range(int(noise_len)):
    X_blobs_noise = np.append(X_blobs_noise, [[random.uniform(0, 20), random.uniform(0, 20)]], axis=0)

plt.scatter(X_blobs_noise[:,0],X_blobs_noise[:,1])
plt.title("Сгенерированное множество с аномалиями")
plt.show()

# 2 Набор: Ирисы Фишера
iris_data = load_iris().data
iris_data = np.delete(iris_data, [2, 3], 1)   #Удаляем излишние столбцы

plt.scatter(iris_data[:,0],iris_data[:,1])
plt.title("Ирисы фишера")
plt.show()


# Метод вложенных циклов
findByNestedLoops(X_blobs_noise, 3, 0.1)

findByNestedLoops(iris_data, 0.9, 0.1)

# Кластеризация
model=DBSCAN(eps=2, min_samples=3)
model.fit(X_blobs_noise)
plt.scatter(X_blobs_noise[:,0], X_blobs_noise[:,1], c=model.labels_)
plt.title('Нахождение аномалий с помощью плотностной\n кластеризации DBSCAN сгенерированного набора')
plt.show()

model=DBSCAN(eps=0.4, min_samples=3)
model.fit(iris_data)
plt.scatter(iris_data[:,0], iris_data[:,1], c=model.labels_)
plt.title('Нахождение аномалий с помощью плотностной\n кластеризации DBSCAN ирисов Фишера')
plt.show()