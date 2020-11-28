from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
# https://medium.com/data-folks-indonesia/step-by-step-to-understanding-k-means-clustering-and-implementation-with-sklearn-b55803f519d6
# https://www.kaggle.com/arshid/iris-flower-dataset
## Load Data
# dfa = pd.read_csv('../bank/Mall_Customers.csv')
# dfa = dfa[['Age','Annual Income (k$)','Spending Score (1-100)']]
# print('Total Row : ', len(dfa))
# print(dfa)
#
# ## Feature Scaling
# sc_dfa = StandardScaler()
# dfa_std = sc_dfa.fit_transform(dfa.astype(float))
# ## Clustering with KMeans
# kmeans = KMeans(n_clusters=3, random_state=42).fit(dfa_std)
# labels = kmeans.labels_
# new_dfa = pd.DataFrame(data = dfa_std, columns = ['Age','Annual Income (k$)','Spending Score (1-100)'])
# new_dfa['label_kmeans'] = labels
#
# fig = plt.figure(figsize=(20,10))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 0], new_dfa['Annual Income (k$)'][new_dfa.label_kmeans == 0], new_dfa['Spending Score (1-100)'][new_dfa.label_kmeans == 0], c='blue', s=100, edgecolor='green',linestyle='--')
# ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 1], new_dfa['Annual Income (k$)'][new_dfa.label_kmeans == 1], new_dfa['Spending Score (1-100)'][new_dfa.label_kmeans == 1], c='red', s=100, edgecolor='green',linestyle='--')
# ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 2], new_dfa['Annual Income (k$)'][new_dfa.label_kmeans == 2], new_dfa['Spending Score (1-100)'][new_dfa.label_kmeans == 2], c='green', s=100, edgecolor='green',linestyle='--')
# ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 3], new_dfa['Annual Income (k$)'][new_dfa.label_kmeans == 3], new_dfa['Spending Score (1-100)'][new_dfa.label_kmeans == 3], c='orange', s=100, edgecolor='green',linestyle='--')
# ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 4], new_dfa['Annual Income (k$)'][new_dfa.label_kmeans == 4], new_dfa['Spending Score (1-100)'][new_dfa.label_kmeans == 4], c='purple', s=100, edgecolor='green',linestyle='--')
# centers = kmeans.cluster_centers_
# ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=500);
# plt.xlabel('Age')
# plt.ylabel('Annual Income (k$)')
# ax.set_zlabel('Spending Score (1-100)')
# plt.show()

## Load Data
dfa = pd.read_csv("../bank/Mall_Customers.csv")
dfa = dfa[['Age','Annual Income (k$)']]
print('Total Row : ', len(dfa))
## Feature Scaling
sc_dfa = StandardScaler()
dfa_std = sc_dfa.fit_transform(dfa.astype(float))
## Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42).fit(dfa_std)
labels = kmeans.labels_
new_dfa = pd.DataFrame(data = dfa_std, columns = ['Age','Annual Income (k$)'])
new_dfa['label_kmeans'] = labels
fig, ax = plt.subplots(figsize=(10,7))
plt.scatter(new_dfa["Annual Income (k$)"][new_dfa["label_kmeans"] == 0], new_dfa["Age"][new_dfa["label_kmeans"] == 0],
            color = "blue", s=100, edgecolor='green',linestyle='--')
plt.scatter(new_dfa["Annual Income (k$)"][new_dfa["label_kmeans"] == 1], new_dfa["Age"][new_dfa["label_kmeans"] == 1],
            color = "red", s=100, edgecolor='green',linestyle='--')
plt.scatter(new_dfa["Annual Income (k$)"][new_dfa["label_kmeans"] == 2], new_dfa["Age"][new_dfa["label_kmeans"] == 2],
            color = "green", s=100, edgecolor='green',linestyle='--')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=500)
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Age')
plt.show()