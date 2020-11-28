from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

# https://saskeli.github.io/data-analysis-with-python-summer-2019/clustering.html
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

noise_1_len = 0.01 * len(y_blobs)

for i in range(int(noise_1_len)):
    # to do
    pass

X,y = make_moons(200, noise=0.05, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()

# Выполним зашумление датасета
x = random.uniform(10.5, 100.5)
