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

# Нахождение частых наборов для датасета Палаты представителей
# Цель: нахождение "любимой" партии за последние 50 лет по палатам представителей
# О

house_candidates_list = []

with open('datasets/1976-2018-house3.csv', 'r', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    house_candidates_list = list(reader)
    house_candidates_list.pop(0)

house_candidates_list = np.delete(house_candidates_list, list(range(2, 12)) + list(range(13, 15)) + list(range(16, 20)), 1)   #Удаляем излишние столбцы

print(house_candidates_list[:10])

current_state_year = []
for candidate in house_candidates_list:

    if current_state_year and current_state_year[-1][0] == candidate[0] and current_state_year[-1][1] == candidate[1]:
        current_state_year.append(candidate)
    else:
        if current_state_year:
            current_min = -1
            current_winners = []
            for candidate_in_year in current_state_year:
                if len(current_winners) < 3:
                    current_winners.append(candidate_in_year)
                    if len(current_winners) == 0:
                        current_min = int(candidate_in_year[3])
                    else:
                        current_min = int(min(current_winners, key=lambda x: int(x[3]))[3])
                elif int(candidate_in_year[3]) > current_min:
                    lambda_min = lambda element: int(element[3]) == current_min
                    current_winners = list(filter(lambda_min, current_winners)).append(candidate_in_year) ## Тут я нахожу просто минимальный элемент, а не удаляю его
                    current_min = int(min(current_winners, key=lambda x: int(x[3]))[3])
            print(current_winners)
            break
        else:
            current_state_year.append(candidate)





