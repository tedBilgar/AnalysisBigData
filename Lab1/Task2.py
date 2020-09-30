import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import mlxtend.frequent_patterns
from utils import eclat2
import matplotlib.pyplot as plt
import time
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

vg_list = []
us_cars_list = []
netflix_list = []

# Получение записей видеоигр
vg_df = pd.read_csv('../bank/vgsales.csv')
print(vg_df[:5])
vg_list = vg_df.drop(['Rank', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1).astype(str).values.tolist()

# Получение Машин США https://www.kaggle.com/doaaalsenani/usa-cers-dataset
us_cars_df = pd.read_csv('../bank/us_cars.csv')
print(us_cars_df[:5])
us_cars_list = us_cars_df.drop([us_cars_df.columns[0], 'mileage', 'vin', 'lot', 'country', 'model', 'title_status'], axis=1).astype(str).values.tolist()

# Получение записей Netflix
netflix_df = pd.read_csv('../bank/netflix_titles.csv')
netflix_df["director"].fillna("No Director", inplace=True)
netflix_list = netflix_df.astype(str).values.tolist()

# Начало трансформации частых наборов
te = TransactionEncoder()
vg_te_ary = te.fit(vg_list).transform(vg_list)
vg_df = pd.DataFrame(vg_te_ary, columns=te.columns_)

us_cars_te_ary = te.fit(us_cars_list).transform(us_cars_list)
us_cars_df = pd.DataFrame(us_cars_te_ary, columns=te.columns_)

netflix_te_ary = te.fit(netflix_list).transform(netflix_list)
netflix_df = pd.DataFrame(netflix_te_ary, columns=te.columns_)


# Apriori
print('Apriori-Video Games')
vg_freq_items = apriori(vg_df, min_support=0.002, use_colnames=True).sort_values(by='support', ascending=False)
print(vg_freq_items)
print('Apriori-US Cars')
start_time = time.time()
us_cars_freq_items = apriori(us_cars_df, min_support=0.02, use_colnames=True).sort_values(by='support', ascending=False)
print(us_cars_freq_items)
apriori_time = time.time() - start_time
print('Apriori-Netflix')
netflix_freq_items = apriori(netflix_df, min_support=0.08, use_colnames=True).sort_values(by='support', ascending=False)
print(netflix_freq_items)

# Получение правил
vg_time_list = []
vg_rules_count_list = []
vg_max_items_list = []
vg_no_more_seven_list = []

print('Association rules for videogames (support=0,2%, confidence=60%):')
start_time = time.time()
vg_rules = association_rules(vg_freq_items, metric="confidence", min_threshold=0.6).sort_values(by='confidence', ascending=False)
vg_time_list.append((time.time() - start_time))
vg_rules_count_list.append(len(vg_rules.index))
vg_object_list = list(map(lambda x: (len(x[0]) + len(x[1])), vg_rules[['antecedents', 'consequents']].values.tolist()))
vg_max_items_list.append(max(vg_object_list))
vg_no_more_seven_list.append(len(list(filter(lambda x: x < 7, vg_object_list))))
print(vg_rules)

print('Association rules for videogames (support=0,2%, confidence=65%):')
start_time = time.time()
vg_rules = association_rules(vg_freq_items, metric="confidence", min_threshold=0.65).sort_values(by='confidence', ascending=False)
vg_time_list.append((time.time() - start_time))
vg_rules_count_list.append(len(vg_rules.index))
vg_object_list = list(map(lambda x: (len(x[0]) + len(x[1])), vg_rules[['antecedents', 'consequents']].values.tolist()))
vg_max_items_list.append(max(vg_object_list))
vg_no_more_seven_list.append(len(list(filter(lambda x: x < 7, vg_object_list))))
print(vg_rules)

print('Association rules for videogames (support=0,2%, confidence=70%):')
start_time = time.time()
vg_rules = association_rules(vg_freq_items, metric="confidence", min_threshold=0.7).sort_values(by='confidence', ascending=False)
vg_time_list.append((time.time() - start_time))
vg_rules_count_list.append(len(vg_rules.index))
vg_object_list = list(map(lambda x: (len(x[0]) + len(x[1])), vg_rules[['antecedents', 'consequents']].values.tolist()))
vg_max_items_list.append(max(vg_object_list))
vg_no_more_seven_list.append(len(list(filter(lambda x: x < 7, vg_object_list))))
print(vg_rules)

print('Association rules for videogames (support=0,2%, confidence=75%):')
start_time = time.time()
vg_rules = association_rules(vg_freq_items, metric="confidence", min_threshold=0.75).sort_values(by='confidence', ascending=False)
vg_time_list.append((time.time() - start_time))
vg_rules_count_list.append(len(vg_rules.index))
vg_object_list = list(map(lambda x: (len(x[0]) + len(x[1])), vg_rules[['antecedents', 'consequents']].values.tolist()))
vg_max_items_list.append(max(vg_object_list))
vg_no_more_seven_list.append(len(list(filter(lambda x: x < 7, vg_object_list))))
print(vg_rules)

print('Association rules for US Cars (support=2%, confidence=60%):')
us_cars_rules = association_rules(us_cars_freq_items, metric="confidence", min_threshold=0.6).sort_values(by='confidence', ascending=False)
print(us_cars_rules)

print('Association rules for US Cars (support=2%, confidence=65%):')
us_cars_rules = association_rules(us_cars_freq_items, metric="confidence", min_threshold=0.65).sort_values(by='confidence', ascending=False)
print(us_cars_rules)

print('Association rules for US Cars (support=2%, confidence=70%):')
us_cars_rules = association_rules(us_cars_freq_items, metric="confidence", min_threshold=0.7).sort_values(by='confidence', ascending=False)
print(us_cars_rules)

print('Association rules for US Cars (support=2%, confidence=75%):')
us_cars_rules = association_rules(us_cars_freq_items, metric="confidence", min_threshold=0.75).sort_values(by='confidence', ascending=False)
print(us_cars_rules)

print('Association rules for Netflix (support=5%, confidence=60%):')
netflix_rules = association_rules(netflix_freq_items, metric="confidence", min_threshold=0.6).sort_values(by='confidence', ascending=False)
print(netflix_rules)

print('Association rules for Netflix (support=5%, confidence=65%):')
netflix_rules = association_rules(netflix_freq_items, metric="confidence", min_threshold=0.65).sort_values(by='confidence', ascending=False)
print(netflix_rules)

print('Association rules for Netflix (support=5%, confidence=70%):')
netflix_rules = association_rules(netflix_freq_items, metric="confidence", min_threshold=0.7).sort_values(by='confidence', ascending=False)
print(netflix_rules)

print('Association rules for Netflix (support=5%, confidence=75%):')
netflix_rules = association_rules(netflix_freq_items, metric="confidence", min_threshold=0.75).sort_values(by='confidence', ascending=False)
print(netflix_rules)

# Визуализация данных
# Время выполнения алгоритмов
plt.bar(['60', '65', '70', '75'], vg_time_list)
plt.title('Время выполнения')
plt.ylabel('Время выполнения, мс')
plt.xlabel('Уровень достоверности, %')
plt.show()

# общее количество найденных правил
plt.bar(['60', '65', '70', '75'], vg_rules_count_list)
plt.title('Общее количество найденных правил')
plt.ylabel('Количество найденных правил')
plt.xlabel('Уровень достоверности, %')
plt.show()

# максимальное количество объектов в правиле
plt.bar(['60', '65', '70', '75'], vg_max_items_list)
plt.title('Максимальное количество объектов в правиле')
plt.ylabel('Количество объектов в правиле')
plt.xlabel('Уровень достоверности, %')
plt.show()

# количество правил, в которых антецедент и консеквент суммарно
# включают в себя не более семи объектов
plt.bar(['60', '65', '70', '75'], vg_no_more_seven_list)
plt.title('Количество правил, в которых антецедент и консеквент\n суммарно включают в себя не более семи объектов', fontdict={'fontsize': 10})
plt.ylabel('Количество правил, в которых антецедент и консеквент\n суммарно включают в себя не более семи объектов', fontdict={'fontsize': 10})
plt.xlabel('Уровень достоверности, %')
plt.show()
