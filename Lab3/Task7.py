import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage


noisy_circles = datasets.make_circles(n_samples=100, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(100, noise=0.05, random_state=0)
blobs = datasets.make_blobs(centers=4, n_samples=100, random_state=0, cluster_std=0.7)

dataset_list = [
    (noisy_circles, {'n_clusters': 2}, 'Circles'),
    (noisy_moons, {'n_clusters': 2}, 'Moons'),
    (blobs, {'n_clusters': 4}, 'Blobs')
]
for (dataset, alg_params, data_name) in dataset_list:
    params = alg_params
    X, y = dataset
    ward = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward')
    complete = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='complete')
    average = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='average')
    single = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='single')

    clustering_algorithms = (
            ('Single', single),
            ('Average', average),
            ('Complete', complete),
            ('Ward', ward),
        )

    for name, algorithm in clustering_algorithms:
        algorithm.fit(X)

        plt.scatter(X[:, 0], X[:, 1], c=algorithm.labels_)
        plt.title('Визуализация для датасета ' + str(data_name) + ' Linkage алгоритмом ' + str(name))
        plt.show()

        Z = linkage(X, method=str(name).lower())
        dendrogram(Z)
        plt.show()

