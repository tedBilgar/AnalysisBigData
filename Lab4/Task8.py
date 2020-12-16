import random
from scipy.stats import chisquare
import numpy as np
from matplotlib import pyplot as plt

int_list = []
for i in range(0, 5000):
    int_list.append(random.randint(0,100))

# Добавление шума в одномерном наборе
for i in range(0, 10):
    int_list[random.randint(0,5000)] = random.randint(100, 500)


dictionary = {1: 4, 2: 1, 3: 10}
plt.bar(list(dictionary.keys()), dictionary.values(), color='g')
plt.show()