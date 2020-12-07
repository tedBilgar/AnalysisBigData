import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori, association_rules
from utils import eclat2
import matplotlib.pyplot as plt
import time
import numpy as np
from functools import reduce

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# Нахождение поддержки со стороны сомневающихся штатов

poll_2020_list = []

with open('datasets/presidential_polls_2020.csv', 'r') as f:
    reader = csv.reader(f)
    poll_2020_list = list(reader)
    poll_2020_list.pop(0)

poll_2020_list = np.delete(poll_2020_list, list(range(0, 1)) + list(range(2, 3)) + list(range(4, 11)) + list(range(12, 17)), 1)   #Удаляем излишние столбцы

poll_2020_list_refactor = []

for poll in poll_2020_list:
    poll_2020_list_refactor.append([poll[0], poll[1], float(poll[2])])

print(poll_2020_list_refactor[:10])


def get_sum_votes(poll_list, state, candidate_name):
    state_candidate_poll_list = list(filter(lambda x: x[0] == state and x[1] == candidate_name, poll_list))

    state_candidate_poll_list_sum = 0

    for poll in state_candidate_poll_list:
        state_candidate_poll_list_sum += poll[2]

    return state_candidate_poll_list_sum


def print_votes(poll_list, state, candidate_name):
    print('Общий процентаж для ' + str(candidate_name) + ' в штате ' + str(state) + ': ' + str(get_sum_votes(poll_list, state, candidate_name)))


print_votes(poll_2020_list_refactor, 'Colorado', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'Colorado', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'Vermont', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'Vermont', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'Illinois', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'Illinois', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'West Virginia', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'West Virginia', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'New Hampshire', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'New Hampshire', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'Ohio', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'Ohio', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'Michigan', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'Michigan', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'Nevada', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'Nevada', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'New Mexico', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'New Mexico', 'Donald Trump')

print_votes(poll_2020_list_refactor, 'Iowa', 'Joseph R. Biden Jr.')
print_votes(poll_2020_list_refactor, 'Iowa', 'Donald Trump')
