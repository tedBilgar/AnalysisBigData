from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from sklearn.cluster import DBSCAN

# Выполним кластеризацию с невыпуклыми кластерами
# X_moons,y_moons = make_moons(200, noise=0.05, random_state=0)
# plt.scatter(X_moons[:,0], X_moons[:,1])
# plt.show()
#
# for min_pts in range(3, 10):
#     model=DBSCAN(eps=0.2, min_samples=min_pts)
#     model.fit(X_moons)
#     plt.scatter(X_moons[:,0], X_moons[:,1], c=model.labels_)
#     plt.title('Вcего ' + str(len(set(model.labels_))) + ' кластера/ов. Eps=0.2, minPts=' + str(min_pts))
#     plt.show()
#
# for eps in [0.1, 0.2, 0.5, 1, 2]:
#     model=DBSCAN(eps=eps, min_samples=5)
#     model.fit(X_moons)
#     plt.scatter(X_moons[:,0], X_moons[:,1], c=model.labels_)
#     plt.title('Вcего ' + str(len(set(model.labels_))) + ' кластера/ов. Eps=' + str(eps) + ', minPts=5')
#     plt.show()


# Выполнение плотностной кластеризации с выпуклыми кластерами
X_blobs,y_blobs = make_blobs(centers=4, n_samples=400, random_state=0, cluster_std=0.5)
plt.scatter(X_blobs[:,0],X_blobs[:,1])
plt.show()
noise_len = 0.1 * len(y_blobs)
X_blobs_noise = X_blobs.copy()
for i in range(int(noise_len)):
    X_blobs_noise = np.append(X_blobs_noise, [[random.uniform(0, 20), random.uniform(0, 20)]], axis=0)

for min_pts in range(3, 10):
    model=DBSCAN(eps=0.5, min_samples=min_pts)
    model.fit(X_blobs_noise)
    plt.scatter(X_blobs_noise[:,0], X_blobs_noise[:,1], c=model.labels_)
    plt.title('Вcего ' + str(len(set(model.labels_))) + ' кластера/ов. Eps=0.2, minPts=' + str(min_pts))
    plt.show()

# for eps in [0.1, 0.2, 0.5, 1, 2]:
#     model=DBSCAN(eps=eps, min_samples=5)
#     model.fit(X_blobs)
#     plt.scatter(X_blobs[:,0], X_blobs[:,1], c=model.labels_)
#     plt.title('Вcего ' + str(len(set(model.labels_))) + ' кластера/ов. Eps=' + str(eps) + ', minPts=5')
#     plt.show()

