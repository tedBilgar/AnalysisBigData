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
vg_list = vg_df.drop(['Rank', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1).astype(str).values.tolist()

# Получение Машин США
us_cars_df = pd.read_csv('../bank/us_cars.csv')
us_cars_list = us_cars_df.drop([us_cars_df.columns[0], 'mileage', 'vin', 'lot'], axis=1).astype(str).values.tolist()

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
vg_freq_items = apriori(vg_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False)
print(vg_freq_items)
print('Apriori-US votes')
start_time = time.time()
us_cars_freq_items = apriori(us_cars_df, min_support=0.01, use_colnames=True).sort_values(by='support', ascending=False)
print(us_cars_freq_items)
apriori_time = time.time() - start_time
print('Apriori-Netflix')
netflix_freq_items = apriori(netflix_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False)
print(netflix_freq_items)

print('Association rules for videogames:')
vg_rules = association_rules(vg_freq_items, metric="confidence", min_threshold=0.6)
print(vg_rules)

print('Association rules for US Cars:')
us_cars_rules = association_rules(us_cars_freq_items, metric="confidence", min_threshold=0.6)
print(us_cars_rules)

print('Association rules for Netflix:')
netflix_rules = association_rules(netflix_freq_items, metric="confidence", min_threshold=0.6)
print(netflix_rules)
