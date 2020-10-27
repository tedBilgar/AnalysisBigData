from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import tree
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

mob_df = pd.read_csv('../test_bank/mob_price_classes.csv')
feature_cols = list(mob_df.columns)[:-1]

i = 1
clf = DecisionTreeClassifier()

while i <= 1:
    mob_train, mob_test = train_test_split(mob_df, test_size=round(0.4 - 0.05 * i, 2), shuffle=True)

    mob_train_features = mob_train.drop(['price_range'], axis=1).values.tolist()
    mob_train_labels = mob_train['price_range'].values.tolist()
    mob_test_features = mob_test.drop(['price_range'], axis=1).values.tolist()
    mob_test_labels = mob_test['price_range'].values.tolist()

    clf = clf.fit(mob_train_features, mob_train_labels)

    y_pred = clf.predict(mob_test_features)

    print("Accuracy:", metrics.accuracy_score(mob_test_labels, y_pred))

    tree.plot_tree(clf)

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('diabetes.png')
    Image(graph.create_png())

    i = i + 1

