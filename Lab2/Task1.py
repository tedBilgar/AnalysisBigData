from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
import pandas as pd

wine_list = []

wine_df = pd.read_csv('../bank/wine.csv')

wine_train, wine_test = train_test_split(wine_df, test_size=0.05, shuffle=True)

wine_train_features = wine_train.drop(['quality'], axis=1).values.tolist()
wine_train_labels = wine_train['quality'].values.tolist()

wine_test_features = wine_test.drop(['quality'], axis=1).values.tolist()
wine_test_labels = wine_test['quality'].values.tolist()

model = GaussianNB()

summary = model.fit(wine_train_features, wine_train_labels)

wine_predict = model.predict(wine_test_features)

print("Accuracy:", metrics.accuracy_score(wine_test_labels, wine_predict))
print("Precision:", metrics.precision_score(wine_test_labels, wine_predict, average='micro'))
print("Recall:", metrics.recall_score(wine_test_labels, wine_predict, average='micro'))
