import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

##
## Данный скрипт предназначен для изучения точности и важности опросов от определенных газет журналов и источников
##
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

president_poll_list = []
## https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VWXHQZ
with open('datasets/president_polls_data2016.csv', 'r', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    president_poll_list = list(reader)
    president_poll_list.pop(0)

president_poll_list = np.delete(president_poll_list, [0, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] , 1)   #Удаляем излишние столбцы

print(president_poll_list[0])
president_poll_list_winner = []

for poll in president_poll_list:
    if poll[2] != 'NA' and int(poll[2]) > int(poll[3]):
        president_poll_list_winner.append(np.append(poll, 'Trump'))
    else:
        president_poll_list_winner.append(np.append(poll, 'Clinton'))

president_poll_list_winner = np.delete(president_poll_list_winner, [2, 3] , 1)   #Удаляем излишние столбцы со значением голосов, тк данная информация уже не нужна

print(president_poll_list_winner[:10])

# Выполняем нахождение частых наборов по "Штат", "партия"
te = TransactionEncoder()
president_poll_te_ary = te.fit(president_poll_list_winner).transform(president_poll_list_winner)
president_poll_df = pd.DataFrame(president_poll_te_ary, columns=te.columns_)

print('Apriori-Winners')
# Минимум 100 элементов по частоте => 0,03
president_poll_freq_items = apriori(president_poll_df, min_support=0.03, use_colnames=True).sort_values(by='support', ascending=False)
print(president_poll_freq_items)

print('Association rules for Winners (support=0,03%, confidence=40%):')
winners_rules = association_rules(president_poll_freq_items, metric="confidence", min_threshold=0.4).sort_values(by='confidence', ascending=False)

print(winners_rules)

## Обработка 2020ого
# https://data.fivethirtyeight.com/
# https://github.com/fivethirtyeight/data/tree/master/polls

with open('datasets/president_approval_polls_2020.csv', 'r', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    president_poll_list = list(reader)
    president_poll_list.pop(0)

president_poll_2020_list = np.delete(president_poll_list, list(range(0, 6)) + list(range(7, 16)) + list(range(17, 25)), 1)   #Удаляем излишние столбцы
print(president_poll_2020_list[0])

president_poll_2020_list_winner = []

for poll in president_poll_2020_list:
    if poll[2] != 'NA' and float(poll[2]) > float(poll[3]):
        president_poll_2020_list_winner.append(np.append(poll, 'Trump'))
    else:
        president_poll_2020_list_winner.append(np.append(poll, 'Biden'))

president_poll_2020_list_winner = np.delete(president_poll_2020_list_winner, [2, 3] , 1)
print(president_poll_2020_list_winner[:2])

te = TransactionEncoder()
president_2020_poll_te_ary = te.fit(president_poll_2020_list_winner).transform(president_poll_2020_list_winner)
president_2020_poll_df = pd.DataFrame(president_2020_poll_te_ary, columns=te.columns_)

print('Apriori-Winners')
# Минимум 100 элементов по частоте => 0,03
president_2020_poll_freq_items = apriori(president_2020_poll_df, min_support=0.01, use_colnames=True).sort_values(by='support', ascending=False)
print(president_2020_poll_freq_items)

print('Association rules for Winners (support=0,01%, confidence=40%):')
winners_2020_rules = association_rules(president_2020_poll_freq_items, metric="confidence", min_threshold=0.4).sort_values(by='confidence', ascending=False)

print(winners_2020_rules)
