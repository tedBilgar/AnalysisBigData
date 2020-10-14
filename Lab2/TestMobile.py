from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

mob_df = pd.read_csv('../test_bank/mob_price_classes.csv')

i = 1
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

while i <= 7:
    mob_train, mob_test = train_test_split(mob_df, test_size=round(0.4 - 0.05 * i, 2), shuffle=True)

    mob_train_features = mob_train.drop(['price_range'], axis=1).values.tolist()
    mob_train_labels = mob_train['price_range'].values.tolist()
    mob_test_features = mob_test.drop(['price_range'], axis=1).values.tolist()
    mob_test_labels = mob_test['price_range'].values.tolist()

    model = GaussianNB()

    summary = model.fit(mob_train_features, mob_train_labels)

    mob_predict = model.predict(mob_test_features)

    accuracy_list.append(metrics.accuracy_score(mob_test_labels, mob_predict))
    precision_list.append(metrics.precision_score(mob_test_labels, mob_predict, average='weighted'))
    recall_list.append(metrics.recall_score(mob_test_labels, mob_predict, average='weighted'))
    f1_list.append(metrics.f1_score(mob_test_labels, mob_predict, average='weighted'))

    print("Accuracy:", accuracy_list[-1])
    print("Precision:", precision_list[-1])
    print("Recall:", recall_list[-1])
    print("F-score:", f1_list[-1])

    i = i + 1

print(accuracy_list)

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
