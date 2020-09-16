import csv
from efficient_apriori import apriori
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import mlxtend.frequent_patterns
from utils import eclat



bank_list = []
financial_list = []
forest_fires_list = []

with open('../bank/bank.csv', 'r') as f:
    reader = csv.reader(f)
    bank_list = list(reader)
    bank_list.pop(0)

with open('../bank/financial_stat.csv', 'r') as f:
    reader = csv.reader(f)
    financial_list = list(reader)
    financial_list.pop(0)


with open('../bank/forestfires.csv', 'r') as f:
    reader = csv.reader(f)
    forest_fires_list = list(reader)
    forest_fires_list.pop(0)

# fin_itemsets, fin_rules = efficient_apriori.apriori(financial_list, min_support=0.1)
# #print(fin_rules)
#
# # FP-GROWTH
# te = TransactionEncoder()
# te_ary = te.fit(dataset).transform(dataset)
# df = pd.DataFrame(te_ary, columns=te.columns_)
# print(df)
#
# print(fpgrowth(df, min_support=0.6, use_colnames=True))

te = TransactionEncoder()
fin_te_ary = te.fit(financial_list).transform(financial_list)
fin_df = pd.DataFrame(fin_te_ary, columns=te.columns_)

bank_te_ary = te.fit(bank_list).transform(bank_list)
bank_df = pd.DataFrame(bank_te_ary, columns=te.columns_)

fire_te_ary = te.fit(forest_fires_list).transform(forest_fires_list)
fire_df = pd.DataFrame(fire_te_ary, columns=te.columns_)

# Apriori
print('Apriori-Finance')
print(mlxtend.frequent_patterns.apriori(fin_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
print('Apriori-Bank')
print(mlxtend.frequent_patterns.apriori(bank_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
print('Apriori-Fire')
print(mlxtend.frequent_patterns.apriori(fire_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))

#FP-GROWTH
print('FP-GROWTH-Finance')
print(mlxtend.frequent_patterns.apriori(fin_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
print('FP-GROWTH-Bank')
print(mlxtend.frequent_patterns.apriori(bank_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
print('FP-GROWTH-Fire')
print(mlxtend.frequent_patterns.apriori(fire_df, min_support=0.1, use_colnames=True).sort_values(by='support', ascending=False))
