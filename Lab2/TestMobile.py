from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
import pandas as pd

mob_df = pd.read_csv('../test_bank/mob_train.csv')

mob_train, mob_test = train_test_split(mob_df, test_size=0.1, shuffle=True)


mob_train_features = mob_train.drop(['price_range'], axis=1).values.tolist()
mob_train_labels = mob_train['price_range'].values.tolist()

mob_test_features = mob_test.drop(['price_range'], axis=1).values.tolist()
mob_test_labels = mob_test['price_range'].values.tolist()

model = GaussianNB()

summary = model.fit(mob_train_features, mob_train_labels)

mob_predict = model.predict(mob_test_features)

print("Accuracy:", metrics.accuracy_score(mob_test_labels, mob_predict))
print("Precision:", metrics.precision_score(mob_test_labels, mob_predict, average='micro'))
print("Recall:", metrics.recall_score(mob_test_labels, mob_predict, average='micro'))