import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori, association_rules
from utils import eclat2
import matplotlib.pyplot as plt
import time
import numpy as np
from operator import itemgetter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# Нахождение частых наборов для датасета Палаты представителей
# Цель: нахождение "любимой" партии за последние 50 лет по палатам представителей

house_candidates_list = []

with open('datasets/1976-2018-house3.csv', 'r', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    house_candidates_list = list(reader)
    house_candidates_list.pop(0)

house_candidates_list = np.delete(house_candidates_list, list(range(2, 12)) + list(range(13, 15)) + list(range(16, 20)), 1)   #Удаляем излишние столбцы

house_candidates_list = house_candidates_list[15418::]

house_pref_list = []
current_state_year = []
current_min = -1
for candidate in house_candidates_list:

    if current_state_year and current_state_year[-1][0] == candidate[0] and current_state_year[-1][1] == candidate[1]:
        current_state_year.append(candidate)
    else:
        if current_state_year:
            current_winners = []
            for candidate_in_year in current_state_year:
                candidate_in_year = candidate_in_year.tolist()
                if len(current_winners) < 3:
                    candidate_in_year[3] = int(candidate_in_year[3])
                    current_winners.append(candidate_in_year)
                    if len(current_winners) == 0:
                        current_min = int(candidate_in_year[3])
                    else:
                        current_min = int(min(current_winners, key=lambda x: int(x[3]))[3])
                elif int(candidate_in_year[3]) > current_min:
                    current_winners = sorted(current_winners, key=itemgetter(3), reverse=True)
                    current_winners.pop()
                    candidate_in_year[3] = int(candidate_in_year[3])
                    current_winners.append(candidate_in_year)
                    current_min = int(sorted(current_winners, key=itemgetter(3), reverse=True)[-1][3])
            print(current_winners)
            resp_count = len(list(filter(lambda x: x[2] == 'REPUBLICAN', current_winners)))
            dem_count = len(list(filter(lambda x: x[2] == 'DEMOCRAT', current_winners)))
            current_winners = np.delete(current_winners, [0, 3], 1)
            house_pref_list.extend(current_winners)
            current_state_year = []
            current_state_year.append(candidate)
        else:
            current_state_year.append(candidate)
            current_min = int(candidate[3])

pass

# Выполняем нахождение частых наборов по "Штат", "партия"
te = TransactionEncoder()
house_pref_te_ary = te.fit(house_pref_list).transform(house_pref_list)
house_pref_df = pd.DataFrame(house_pref_te_ary, columns=te.columns_)

print('Apriori-House-Preferences')
house_pref_freq_items = apriori(house_pref_df, min_support=0.01, use_colnames=True).sort_values(by='support', ascending=False)
print(house_pref_freq_items)

print('Association rules for House (support=0,01%, confidence=40%):')
house_pref_rules = association_rules(house_pref_freq_items, metric="confidence", min_threshold=0.4).sort_values(by='confidence', ascending=False)

print(house_pref_rules)


