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

us_list = []
vg_list = []
netflix_list = []

with open('../bank/us.csv', 'r') as f:
    reader = csv.reader(f)
    us_list = list(reader)
    us_list.pop(0)

with open('../bank/vgsales.csv', 'r') as f:
    reader = csv.reader(f)
    vg_list = list(reader)
    vg_list.pop(0)

with open('../bank/netflix_titles.csv', 'r') as f:
    reader = csv.reader(f)
    netflix_list = list(reader)
    netflix_list.pop(0)

# Полезные модификации и трансформации для исходного дата сета
vg_list = np.delete(vg_list, [6, 7, 8, 9, 10], 1) #Удаляем излишние числовые столбцы

#Заполняем пустые места или null значениями
trans_netflix_df = pd.read_csv('../bank/netflix_titles.csv')
trans_netflix_df["director"].fillna("No Director", inplace=True)
trans_netflix_df = trans_netflix_df.astype(str)
netflix_list = trans_netflix_df.values.tolist()

# Начало трансформации частых наборов
te = TransactionEncoder()
vg_te_ary = te.fit(vg_list).transform(vg_list)
vg_df = pd.DataFrame(vg_te_ary, columns=te.columns_)

us_te_ary = te.fit(us_list).transform(us_list)
us_df = pd.DataFrame(us_te_ary, columns=te.columns_)

netflix_te_ary = te.fit(netflix_list).transform(netflix_list)
netflix_df = pd.DataFrame(netflix_te_ary, columns=te.columns_)


# Apriori
print('Apriori-Video Games')
vg_freq_items = apriori(vg_df, min_support=0.001, use_colnames=True).sort_values(by='support', ascending=False)
print(vg_freq_items)
print('Apriori-US votes')
start_time = time.time()
us_freq_items = apriori(us_df, min_support=0.01, use_colnames=True).sort_values(by='support', ascending=False)
print(us_freq_items)
apriori_time = time.time() - start_time
print('Apriori-Netflix')
netflix_freq_items = apriori(netflix_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False)
print(netflix_freq_items)

vg_rules = association_rules(vg_freq_items, metric="confidence", min_threshold=0.6)
print(vg_rules)

us_freq_items = association_rules(us_freq_items, metric="confidence", min_threshold=0.6)
print(us_freq_items)

netflix_rules = association_rules(netflix_freq_items, metric="confidence", min_threshold=0.6)
print(netflix_rules)
