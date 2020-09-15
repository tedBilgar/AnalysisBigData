import csv
import efficient_apriori
import pyfpgrowth

bank_list = []
financial_list = []
forest_fires_list = []

with open('../bank/bank.csv', 'r') as f:
    reader = csv.reader(f)
    bank_list = list(reader)
    bank_list.pop(0)
    print(bank_list)

with open('../bank/financial_stat.csv', 'r') as f:
    reader = csv.reader(f)
    financial_list = list(reader)
    financial_list.pop(0)
    print(financial_list)

with open('../bank/forestfires.csv', 'r') as f:
    reader = csv.reader(f)
    forest_fires_list = list(reader)
    forest_fires_list.pop(0)
    print(forest_fires_list)

fin_itemsets, fin_rules = efficient_apriori.apriori(financial_list, min_support=0.1)
print(fin_rules)

# FP-GROWTH
fin_pattern = pyfpgrowth.find_frequent_patterns(financial_list, 0.1)

rules = pyfpgrowth.generate_association_rules(fin_pattern, 0.1)

print(rules)
