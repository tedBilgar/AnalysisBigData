import random
from matplotlib import pyplot as plt
import pandas as pd
import statistics


def find_outlines_labels_likehood(object_list):
    mean_value = sum(object_list) / len(object_list)
    st_dev = statistics.stdev(object_list)
    outline_labels_list = []
    for elem in object_list:
        if abs(elem - mean_value) / st_dev > 3:
            outline_labels_list.append(1)
        else:
            outline_labels_list.append(0)
    return outline_labels_list


def find_outlines_labels_chi_square(object_list):
    mean_value = sum(object_list) / len(object_list)
    chi_value_list = []

    for elem in object_list:
        chi_value_list.append(((elem - mean_value) ** 2) / mean_value)

    outline_labels_list = []
    for chi_val in chi_value_list:
        if chi_val > mean_value:
            outline_labels_list.append(1)
        else:
            outline_labels_list.append(0)
    return outline_labels_list


# Первый набор составлен самостоятельно и добавлены шумы
elem_size = 100
int_list = []
for i in range(0, elem_size):
    int_list.append(random.randint(0, 100))

# Добавление дополнительного шума в одномерном наборе
for i in range(0, 5):
    int_list[random.randint(0, elem_size)] = random.randint(100, 500)


# Второй набор https://www.kaggle.com/sootersaalu/amazon-top-50-bestselling-books-2009-2019
amazon_df = pd.read_csv('../bank/amazon_bestsellers.csv')
amazon_list = amazon_df['Price'].tolist()


# 1) Метод максимального правдоподобия
out_line_list = find_outlines_labels_likehood(int_list)
plt.scatter(x=int_list, y=[0]*len(int_list), c=out_line_list)
plt.title('Нахождение аномалий в целочисленном наборе данных\nМетод максимального правдоподобия')
plt.show()

out_line_list = find_outlines_labels_likehood(amazon_list)
plt.scatter(x=amazon_list, y=[0]*len(amazon_list), c=out_line_list)
plt.title('Нахождение аномалий в наборе данных Амазон\nМетод максимального правдоподобия')
plt.show()

# 2) Нахождение Хи-Квадрат
out_line_list = find_outlines_labels_chi_square(int_list)
plt.scatter(x=int_list, y=[0]*len(int_list), c=out_line_list)
plt.title('Нахождение аномалий в целочисленном наборе данных\nМетод Хи-квадрат')
plt.show()

out_line_list = find_outlines_labels_chi_square(amazon_list)
plt.scatter(x=amazon_list, y=[0]*len(amazon_list), c=out_line_list)
plt.title('Нахождение аномалий в наборе данных Амазон\nМетод Хи-квадрат')
plt.show()


# 3.1) Нахождение аномалий с помощью гистограмм в одномерном наборе данных целых чисел
int_dictionary = {}
iter_num = 1
for elem in int_list:
    int_dictionary[iter_num] = elem
    iter_num += 1

plt.bar(list(int_dictionary.keys()), int_dictionary.values(), color='g')
plt.title('Гистограмма для нахождения аномалий\n в наборе данных целых чисел')
plt.show()

# 3.2) Нахождение аномалий с помощью гистограмм в одномерном наборе данных продолжительности записи у банка
amazon_dictionary = {}
iter_num = 1
for elem in amazon_list:
    amazon_dictionary[iter_num] = elem
    iter_num += 1

plt.bar(list(amazon_dictionary.keys()), amazon_dictionary.values(), color='g')
plt.title('Гистограмма для нахождения аномалий в наборе стоимости\n бестселлеров Амазона')
plt.show()

