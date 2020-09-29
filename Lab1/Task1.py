import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import mlxtend.frequent_patterns
from utils import eclat2
import matplotlib.pyplot as plt
import time
import numpy as np


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

vg_list = np.delete(vg_list, [6, 7, 8, 9, 10], 1) #Удаляем излишние числовые столбцы

with open('../bank/netflix_titles.csv', 'r') as f:
    reader = csv.reader(f)
    netflix_list = list(reader)
    netflix_list.pop(0)

te = TransactionEncoder()
vg_te_ary = te.fit(vg_list).transform(vg_list)
vg_df = pd.DataFrame(vg_te_ary, columns=te.columns_)

us_te_ary = te.fit(us_list).transform(us_list)
us_df = pd.DataFrame(us_te_ary, columns=te.columns_)

netflix_te_ary = te.fit(netflix_list).transform(netflix_list)
netflix_df = pd.DataFrame(netflix_te_ary, columns=te.columns_)

# Apriori
print('Apriori-Video Games')
print(mlxtend.frequent_patterns.apriori(vg_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
print('Apriori-US votes')
start_time = time.time()
print(mlxtend.frequent_patterns.apriori(us_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
apriori_time = time.time() - start_time;
print('Apriori-Netflix')
print(mlxtend.frequent_patterns.apriori(netflix_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
#
# #FP-GROWTH
print('FP-GROWTH-Video Games')
print(mlxtend.frequent_patterns.apriori(vg_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
print('FP-GROWTH-US votes')
start_time = time.time()
sort_us_df = mlxtend.frequent_patterns.apriori(us_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False)
fp_growth_time = time.time() - start_time
print(sort_us_df)
print('FP-GROWTH-Netflix')
print(mlxtend.frequent_patterns.apriori(netflix_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))

# ECLAT
fin_eclat_res = []
us_eclat_res = []
fire_eclat_res = []
print('ECLAT-Video Games')
print(eclat2.eclat(vg_list, supp=10, out=fin_eclat_res))
start_time = time.time()
print('ECLAT-US votes')
print(eclat2.eclat(us_list, supp=10, out=us_eclat_res))
eclat_time = time.time() - start_time
print('ECLAT-Netflix')
print(eclat2.eclat(netflix_list, supp=10, out=fire_eclat_res))


#Время выполнения алогоритмов
print('Время выполнения алгоритмов')
plt.bar(['Apriori', 'FP-GROWTH', 'ECLAT'], [apriori_time, fp_growth_time, eclat_time])
plt.title('Время выполнения алгоритмов')
plt.ylabel('Время выполнения, мс')
plt.xlabel('Алгоритм')
plt.show()

#Зависимость от поддержки
plt.bar(['10%', '20%', '40%', '60%', '80%'], [
        sort_us_df.loc[sort_us_df['support'] >= 0.1].shape[0],
         sort_us_df.loc[sort_us_df['support'] >= 0.2].shape[0],
          sort_us_df.loc[sort_us_df['support'] >= 0.4].shape[0],
           sort_us_df.loc[sort_us_df['support'] >= 0.6].shape[0],
            sort_us_df.loc[sort_us_df['support'] >= 0.8].shape[0]])
plt.title('Зависимость от поддержки')
plt.ylabel('Количество частых наборов')
plt.xlabel('Уровень поддержки, %')
plt.show()

# Максимальная длина
max_ten = sort_us_df.loc[sort_us_df['support'] >= 0.1]['itemsets'].map(lambda x: len(x)).max()
max_twenty = sort_us_df.loc[sort_us_df['support'] >= 0.2]['itemsets'].map(lambda x: len(x)).max()
max_forty = sort_us_df.loc[sort_us_df['support'] >= 0.4]['itemsets'].map(lambda x: len(x)).max()
max_sixty = sort_us_df.loc[sort_us_df['support'] >= 0.6]['itemsets'].map(lambda x: len(x)).max()
max_eighty = sort_us_df.loc[sort_us_df['support'] >= 0.8]['itemsets'].map(lambda x: len(x)).max()
plt.bar(['10%', '20%', '40%', '60%', '80%'], [
        max_ten,
         max_twenty,
          max_forty,
           max_sixty,
            max_eighty])
plt.title('Зависимость максимальной длины набора')
plt.ylabel('Максимальная длина набора')
plt.xlabel('Уровень поддержки, %')
plt.show()

# Количество наборов различной длины
max_ten = len(set(sort_us_df.loc[sort_us_df['support'] >= 0.1]['itemsets'].map(lambda x: len(x))))
max_twenty = len(set(sort_us_df.loc[sort_us_df['support'] >= 0.2]['itemsets'].map(lambda x: len(x))))
max_forty = len(set(sort_us_df.loc[sort_us_df['support'] >= 0.4]['itemsets'].map(lambda x: len(x))))
max_sixty = len(set(sort_us_df.loc[sort_us_df['support'] >= 0.6]['itemsets'].map(lambda x: len(x))))
max_eighty = len(set(sort_us_df.loc[sort_us_df['support'] >= 0.8]['itemsets'].map(lambda x: len(x))))
plt.bar(['10%', '20%', '40%', '60%', '80%'], [
        max_ten,
         max_twenty,
          max_forty,
           max_sixty,
            max_eighty])
plt.title('Зависимость количества различной длины')
plt.ylabel('Количество частых наборов различной длины')
plt.xlabel('Уровень поддержки, %')
plt.show()