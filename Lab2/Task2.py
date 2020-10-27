from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import tree

mob_df = pd.read_csv('../test_bank/mob_price_classes.csv')
feature_cols = list(mob_df.columns)[:-1]

i = 1
clf = DecisionTreeClassifier()
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

while i <= 1:
    mob_train, mob_test = train_test_split(mob_df, test_size=round(0.4 - 0.05 * i, 2), shuffle=True)

    mob_train_features = mob_train.drop(['price_range'], axis=1).values.tolist()
    mob_train_labels = mob_train['price_range'].values.tolist()
    mob_test_features = mob_test.drop(['price_range'], axis=1).values.tolist()
    mob_test_labels = mob_test['price_range'].values.tolist()

    clf = clf.fit(mob_train_features, mob_train_labels)

    mob_predict = clf.predict(mob_test_features)

    accuracy_list.append(metrics.accuracy_score(mob_test_labels, mob_predict))
    precision_list.append(metrics.precision_score(mob_test_labels, mob_predict, average='weighted'))
    recall_list.append(metrics.recall_score(mob_test_labels, mob_predict, average='weighted'))
    f1_list.append(metrics.f1_score(mob_test_labels, mob_predict, average='weighted'))

    print("Accuracy:", accuracy_list[-1])
    print("Precision:", precision_list[-1])
    print("Recall:", recall_list[-1])
    print("F-score:", f1_list[-1])

    # tree.plot_tree(clf)
    # fig = plt.figure(figsize=(150, 50))
    # _ = tree.plot_tree(clf,
    #                    feature_names=feature_cols,
    #                    class_names=['0', '1', '2', '3'],
    #                    filled=True, fontsize=15)
    #
    # fig.savefig("decistion_tree.png")

    i = i + 1

X = np.arange(start=60, stop=95, step=5)
plt.bar(X - 1.6, accuracy_list, width=0.8, color='b', align='center')
plt.bar(X - 0.8, precision_list, width=0.8, color='g', align='center')
plt.bar(X, recall_list, width=0.8, color='r', align='center')
plt.bar(X + 0.8, f1_list, width=0.8, color='y', align='center')
plt.ylim(0, 1.2)

blue_patch = mpatches.Patch(color='b', label='Accuracy')
green_patch = mpatches.Patch(color='g', label='Precision')
red_patch = mpatches.Patch(color='r', label='Recall')
yellow_patch = mpatches.Patch(color='y', label='F-score')
plt.legend(handles=[blue_patch, green_patch, red_patch, yellow_patch])

plt.title('Метрики классификации')
plt.ylabel('Значение метрики')
plt.xlabel('Мощность обучающей выборки')

plt.show()

