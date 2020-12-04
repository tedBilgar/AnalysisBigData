import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

##
## Данный скрипт предназначен для кластеризации вероятностей, что победит Трамп или Байден
##

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

president_forecasts_list_row = []
# https://data.fivethirtyeight.com/
with open('datasets/presidential_national_toplines_2020.csv', 'r') as f:
    reader = csv.reader(f)
    president_forecasts_list_row = list(reader)
    president_forecasts_list_row.pop(0)

#president_forecasts_list = np.delete(president_forecasts_list, list(range(0, 7)) + list(range(9, 40)), 1)   #Удаляем излишние столбцы
# Анализ шансов Трампа
trump_president_forecasts_list = np.delete(president_forecasts_list_row, list(range(0, 7)) + list(range(8, 14)) + list(range(15, 40)), 1)   #Удаляем излишние столбцы

trump_president_forecasts_list_refactor = []

for poll in trump_president_forecasts_list:
    trump_president_forecasts_list_refactor.append([float(poll[0]), float(poll[1])])

trump_president_forecasts_list_refactor = np.array(trump_president_forecasts_list_refactor)
plt.scatter(trump_president_forecasts_list_refactor[:,0],trump_president_forecasts_list_refactor[:,1])
plt.show()

model = DBSCAN(eps=1, min_samples=3)
model.fit(trump_president_forecasts_list_refactor)
plt.scatter(trump_president_forecasts_list_refactor[:,0], trump_president_forecasts_list_refactor[:,1], c=model.labels_)
plt.title('Кластеризация шансов на победу Трампа.\nВcего ' + str(len(set(model.labels_))) + ' кластера/ов. Eps=1, minPts=3')
plt.show()



# Анализ шансов Байдена
biden_president_forecasts_list = np.delete(president_forecasts_list_row, list(range(0, 8)) + list(range(9, 15)) + list(range(16, 40)), 1)   #Удаляем излишние столбцы

biden_president_forecasts_list_refactor = []

for poll in biden_president_forecasts_list:
    biden_president_forecasts_list_refactor.append([float(poll[0]), float(poll[1])])

biden_president_forecasts_list_refactor = np.array(biden_president_forecasts_list_refactor)
plt.scatter(biden_president_forecasts_list_refactor[:,0],biden_president_forecasts_list_refactor[:,1])
plt.show()

model = DBSCAN(eps=1, min_samples=3)
model.fit(biden_president_forecasts_list_refactor)
plt.scatter(biden_president_forecasts_list_refactor[:,0], biden_president_forecasts_list_refactor[:,1], c=model.labels_)
plt.title('Кластеризация шансов на победу Байдена.\nВcего ' + str(len(set(model.labels_))) + ' кластера/ов. Eps=1, minPts=3')
plt.show()
