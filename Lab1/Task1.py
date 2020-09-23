import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import mlxtend.frequent_patterns
from utils import eclat
from utils import eclat2
import matplotlib.pyplot as plt
import time


us_list = []
financial_list = []
forest_fires_list = []

with open('../bank/us.csv', 'r') as f:
    reader = csv.reader(f)
    us_list = list(reader)
    us_list.pop(0)

with open('../bank/financial_stat.csv', 'r') as f:
    reader = csv.reader(f)
    financial_list = list(reader)
    financial_list.pop(0)


with open('../bank/forestfires.csv', 'r') as f:
    reader = csv.reader(f)
    forest_fires_list = list(reader)
    forest_fires_list.pop(0)

te = TransactionEncoder()
fin_te_ary = te.fit(financial_list).transform(financial_list)
fin_df = pd.DataFrame(fin_te_ary, columns=te.columns_)

us_te_ary = te.fit(us_list).transform(us_list)
us_df = pd.DataFrame(us_te_ary, columns=te.columns_)

fire_te_ary = te.fit(forest_fires_list).transform(forest_fires_list)
fire_df = pd.DataFrame(fire_te_ary, columns=te.columns_)

# Apriori
print('Apriori-Finance')
print(mlxtend.frequent_patterns.apriori(fin_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
print('Apriori-US')
start_time = time.time()
print(mlxtend.frequent_patterns.apriori(us_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
apriori_time = time.time() - start_time;
print('Apriori-Fire')
print(mlxtend.frequent_patterns.apriori(fire_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
#
# #FP-GROWTH
print('FP-GROWTH-Finance')
print(mlxtend.frequent_patterns.apriori(fin_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
print('FP-GROWTH-US')
start_time = time.time()
sort_us_df = mlxtend.frequent_patterns.apriori(us_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False)
fp_growth_time = time.time() - start_time
print(sort_us_df)
print('FP-GROWTH-Fire')
print(mlxtend.frequent_patterns.apriori(fire_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))

# ECLAT
fin_eclat_res = []
us_eclat_res = []
fire_eclat_res = []
print('ECLAT-Fin')
print(eclat2.eclat(financial_list, supp=10, out=fin_eclat_res))
start_time = time.time()
print('ECLAT-US')
print(eclat2.eclat(us_list, supp=10, out=us_eclat_res))
eclat_time = time.time() - start_time
print('ECLAT-Fires')
print(eclat2.eclat(forest_fires_list, supp=10, out=fire_eclat_res))


#Время выполнения алогоритмов
print('Время выполнения алгоритмов')
plt.bar(['Apriori', 'FP-GROWTH', 'ECLAT'], [apriori_time, fp_growth_time, eclat_time])
plt.title('Время выполнения алгоритмов')
plt.show()

#Зависимость от поддержки
plt.bar(['10%', '20%', '40%', '60%', '80%'], [
        sort_us_df.loc[sort_us_df['support'] >= 0.1].shape[0],
         sort_us_df.loc[sort_us_df['support'] >= 0.2].shape[0],
          sort_us_df.loc[sort_us_df['support'] >= 0.4].shape[0],
           sort_us_df.loc[sort_us_df['support'] >= 0.6].shape[0],
            sort_us_df.loc[sort_us_df['support'] >= 0.8].shape[0]])
plt.title('Зависимость от поддержки')
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
plt.show()