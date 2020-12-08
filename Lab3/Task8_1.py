from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Строим выпуклые кластеры 1 Вариант
X_blobs,y_blobs = make_blobs(centers=3, n_samples=200, random_state=0, cluster_std=0.4)
plt.scatter(X_blobs[:,0],X_blobs[:,1])
plt.show()

# 1) Метод локтя
# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

visualizer.fit(X_blobs)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# Строим выпуклые кластеры 2 Вариант
X_blobs,y_blobs = make_blobs(centers=3, n_samples=200, random_state=0, cluster_std=0.8)
plt.scatter(X_blobs[:,0],X_blobs[:,1])
plt.show()

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

visualizer.fit(X_blobs)
visualizer.show()


