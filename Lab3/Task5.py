from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

# Строим выпуклые кластеры
X_blobs,y_blobs = make_blobs(centers=4, n_samples=200, random_state=0, cluster_std=0.7)
plt.scatter(X_blobs[:,0],X_blobs[:,1])
plt.show()

print('Всего ' + str(len(y_blobs)) + ' элементов')

x = 3
while x < 10:
    # Выполняем кластеризацию
    model = KMeans(x)
    model.fit(X_blobs)

    # Выполним визуализацию полученных кластеров
    plt.scatter(X_blobs[:,0], X_blobs[:,1], c=model.labels_)
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, color="red") # Show the centres
    plt.title('Визуализация из ' + str(x) + ' кластеров')
    plt.show()
    x = x + 1

# Добавляем шумы и кластеризуем на 4 центроидах
noise_array = [1, 3, 5, 10]

for noise_percent in noise_array:
    noise_len = noise_percent * 0.01 * len(y_blobs)
    X_blobs_noise = X_blobs.copy()
    for i in range(int(noise_len)):
        X_blobs_noise = np.append(X_blobs_noise, [[random.uniform(0, 20), random.uniform(0, 20)]], axis=0)

    # Выполняем кластеризацию
    for cluster_count in [3, 6, 9]:
        model = KMeans(cluster_count)
        model.fit(X_blobs_noise)

        # Выполним визуализацию полученных кластеров
        plt.scatter(X_blobs_noise[:,0], X_blobs_noise[:,1], c=model.labels_)
        plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, color="red") # Show the centres
        plt.title('Визуализация с шумами ' + str(noise_percent) + '% из ' + str(cluster_count) + ' кластеров')
        plt.show()


# Выполним кластеризацию с невыпуклыми кластерами
X_moons,y_moons = make_moons(200, noise=0.05, random_state=0)
plt.scatter(X_moons[:,0], X_moons[:,1])
plt.show()

for cluster_count in [3, 6, 9]:
    model=KMeans(cluster_count)
    model.fit(X_moons)
    plt.scatter(X_moons[:,0], X_moons[:,1], c=model.labels_)
    plt.title('Визуализация из ' + str(cluster_count) + ' кластеров')
    plt.show()
