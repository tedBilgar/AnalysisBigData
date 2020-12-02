import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

noisy_circles = datasets.make_circles(n_samples=100, factor=.5, noise=.05)

X, y = noisy_circles

Z = linkage(X)

dendrogram(Z)
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=Z.labels_)
plt.show()
