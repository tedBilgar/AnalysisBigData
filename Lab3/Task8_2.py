from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Строим выпуклые кластеры 1 Вариант
X_blobs,y_blobs = make_blobs(centers=3, n_samples=200, random_state=0, cluster_std=0.4)
plt.scatter(X_blobs[:,0],X_blobs[:,1])
plt.show()


# 2) Силуэтный коэффициент
silhouette_score_list = []
for cluster_num in range(2, 10):
    model = KMeans(n_clusters=cluster_num)
    cluster_labels = model.fit_predict(X_blobs)
    silhouette_score_list.append(silhouette_score(X_blobs, cluster_labels))
    if max(silhouette_score_list) == silhouette_score_list[-1]:
        optima_cluster_num = cluster_num
    print('Silhouette Score(n=' + str(cluster_num) + '):' + str(silhouette_score_list[-1]))


plt.clf()
model = KMeans(optima_cluster_num)
model.fit(X_blobs)
plt.scatter(X_blobs[:,0], X_blobs[:,1], c=model.labels_)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, color="red")  # Show the centres
plt.title('Визуализация с оптимальным количеством кластеров - ' + str(optima_cluster_num) + '.\nСилуэтный коэффициент\ncluster_std=0.4')
plt.show()


model = KMeans(3)
model.fit(X_blobs)
plt.scatter(X_blobs[:,0], X_blobs[:,1], c=model.labels_)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, color="red")  # Show the centres
plt.title('Визуализация с оптимальным количеством кластеров - ' + str(optima_cluster_num) + '.\nМетод локтя\ncluster_std=0.4')
plt.show()

# Строим выпуклые кластеры 2 Вариант
X_blobs,y_blobs = make_blobs(centers=3, n_samples=200, random_state=0, cluster_std=0.8)
plt.scatter(X_blobs[:,0],X_blobs[:,1])
plt.show()


# 2) Силуэтный коэффициент
silhouette_score_list = []
for cluster_num in range(2, 10):
    model = KMeans(n_clusters=cluster_num)
    cluster_labels = model.fit_predict(X_blobs)
    silhouette_score_list.append(silhouette_score(X_blobs, cluster_labels))
    if max(silhouette_score_list) == silhouette_score_list[-1]:
        optima_cluster_num = cluster_num
    print('Silhouette Score(n=' + str(cluster_num) + '):' + str(silhouette_score_list[-1]))


plt.clf()
model = KMeans(optima_cluster_num)
model.fit(X_blobs)
plt.scatter(X_blobs[:,0], X_blobs[:,1], c=model.labels_)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, color="red")  # Show the centres
plt.title('Визуализация с оптимальным количеством кластеров - ' + str(optima_cluster_num) + '.\nСилуэтный коэффициент\ncluster_std=0.8')
plt.show()


model = KMeans(3)
model.fit(X_blobs)
plt.scatter(X_blobs[:,0], X_blobs[:,1], c=model.labels_)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, color="red")  # Show the centres
plt.title('Визуализация с оптимальным количеством кластеров - ' + str(optima_cluster_num) + '.\nМетод локтя\ncluster_std=0.8')
plt.show()