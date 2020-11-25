import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori, association_rules
from utils import eclat2
import matplotlib.pyplot as plt
import time
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)


# Нахождение частых наборов для датасета кандидатов в президенты от штатов
# Цель: нахождение "любимой" партии за последние 50 лет у каждого штата

president_candidates_list = []

with open('datasets/president-1976-2016.csv', 'r') as f:
    reader = csv.reader(f)
    president_candidates_list = list(reader)
    president_candidates_list.pop(0)

president_candidates_list = np.delete(president_candidates_list, [2, 3, 4, 5, 7, 9, 12, 13] , 1)   #Удаляем излишние столбцы

#print(president_candidates_list)

winner_list = []

for candidate in president_candidates_list:
    if winner_list:
        if candidate[0] == winner_list[-1][0] and candidate[1] == winner_list[-1][1]:
            if int(candidate[4]) > int(winner_list[-1][4]):
                winner_list[-1] = list(candidate)
        else:
            winner_list.append(candidate)
    else:
        winner_list.append(candidate)

winner_list = np.delete(winner_list, [0, 2, 4, 5] , 1)   #Удаляем излишние столбцы со значением голосов, тк данная информация уже не нужна

print(winner_list)

# Выполняем нахождение частых наборов по "Штат", "Пост", "партия"
te = TransactionEncoder()
winner_te_ary = te.fit(winner_list).transform(winner_list)
winner_df = pd.DataFrame(winner_te_ary, columns=te.columns_)

print('Apriori-Winners')
winners_freq_items = apriori(winner_df, min_support=0.01, use_colnames=True).sort_values(by='support', ascending=False)
print(winners_freq_items)

print('Association rules for Winners (support=0,01%, confidence=60%):')
winners_rules = association_rules(winners_freq_items, metric="confidence", min_threshold=0.4).sort_values(by='confidence', ascending=False)

print(winners_rules)