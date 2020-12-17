import random
from scipy.stats import chisquare
from scipy.stats import rv_continuous
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Первый набор составлен самостоятельно и добавлены шумы
elem_size = 100
int_list = []
for i in range(0, elem_size):
    int_list.append(random.randint(0,100))

# Добавление шума в одномерном наборе
for i in range(0, 5):
    int_list[random.randint(0, elem_size)] = random.randint(100, 500)


# Второй набор https://www.kaggle.com/sootersaalu/amazon-top-50-bestselling-books-2009-2019
amazon_df = pd.read_csv('../bank/amazon_bestsellers.csv')
amazon_list = amazon_df['Price'].tolist()

# 1) Нахождение аномалий с помощью гистограмм в одномерном наборе данных целых чисел
int_dictionary = {}
iter_num = 1
for elem in int_list:
    int_dictionary[iter_num] = elem
    iter_num += 1

plt.bar(list(int_dictionary.keys()), int_dictionary.values(), color='g')
plt.title('Гистограмма для нахождения аномалий\n в наборе данных целых чисел')
plt.show()

# 2) Нахождение аномалий с помощью гистограмм в одномерном наборе данных продолжительности записи у банка
amazon_dictionary = {}
iter_num = 1
for elem in amazon_list:
    amazon_dictionary[iter_num] = elem
    iter_num += 1

plt.bar(list(amazon_dictionary.keys()), amazon_dictionary.values(), color='g')
plt.title('Гистограмма для нахождения аномалий в наборе стоимости\n бестселлеров Амазона')
plt.show()

val = 0 # this is the value where you want the data to appear on the y-axis.
ar = np.arange(10) # just as an example array
plt.plot(int_list, np.zeros_like(int_list) + val, 'x')
plt.show()

mle = rv_continuous.fit(data=int_list)
print(mle)
