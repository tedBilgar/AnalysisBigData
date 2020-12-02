import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage


noisy_circles = datasets.make_circles(n_samples=1500, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(200, noise=0.05, random_state=0)
blobs = datasets.make_blobs(centers=4, n_samples=200, random_state=0, cluster_std=0.7)

dataset_list = [
    (noisy_circles, {'n_clusters': 2}, 'Circles'),
    (noisy_moons, {'n_clusters': 2}, 'Moons'),
    (blobs, {'n_clusters': 4}, 'Blobs')
]

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

for (dataset, alg_params, data_name) in dataset_list:
    params = alg_params
    X, y = dataset
    ward = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward')
    complete = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='complete')
    average = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='average')
    single = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='single')

    clustering_algorithms = (
            ('Single Linkage', single),
            ('Average Linkage', average),
            ('Complete Linkage', complete),
            ('Ward Linkage', ward),
        )

    for name, algorithm in clustering_algorithms:
        algorithm.fit(X)

        plt.scatter(X[:, 0], X[:, 1], c=algorithm.labels_)
        plt.title('Визуализация для датасета ' + str(data_name) + ' алгоритмом ' + str(name))
        plt.show()

        plot_dendrogram(algorithm, truncate_mode='level', p=3)

