# Инцидент №1. Отказ оборудования

'''
Входные данные:

1. Время эксплуатации (time_operating)
2. Год введения в эксплутацию (year_operating)
3. Сколько раз оборудование ломалось? (breaking) - поломка
4. Сколько раз его чинили? (repair) - ремонт
5. Сколько раз оборудование осматривали? (review) - смотр

Результат: Вероятность поломки оборудования высока?

breaking = 26
repair = 24
review = 18'''

# Начнем с простой сети: вероятность поломки оборудования, используя только один набор данных
# используя библиотеку NumPy. Взвешенная сумма (три входа - один выход)
'''
import numpy as np

weights = np.array([0.1, 0.2, 0.0])

breaking = np.array([0, 0, 0, 1, 2, 1, 3, 0, 0, 1, 4]) # Количество поломок оборудования за 11 лет
repair = np.array([0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 3]) # Сколько раз оборудование ремонтировалось
review = np.array([1, 1, 2, 0, 0, 0, 0, 3, 1, 2, 0]) # Сколько раз оборудование было осмотрено

_input = np.array([breaking[10], repair[10], review[10]]) # 11 год эксплуатации

def w_sum(a,b):
    assert(len(a) == len(b))
    output = 0

    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

# функция dot задает сумму весов

def neural_network(_input, weights):
    pred = _input.dot(weights)
    return pred

pred = neural_network(_input, weights)

print(pred) # Выведет 1.0 - стопроцентный прогноз поломки с учетом входных данных
'''

# Вариант №2. Один вход - несколько выходов. Способ в данном случае не очень подходит.

'''weights = [0.1, 0.2, 0.0]

breaking = [0, 0, 0, 1, 2, 1, 3, 0, 0, 1, 4] # Количество поломок оборудования за 11 лет

_input = breaking[10]

def ele_mul(number, vector):
    output = [0, 0, 0]

    assert(len(output) == len(vector))

    for i in range(len(vector)):
        output[i] = number * vector[i]
    
    return output

def neural_network(_input, weights):
    pred = ele_mul(_input, weights)

    return pred

pred = neural_network(_input, weights)
print(pred)'''

# Вариант №3. Несколько входов - несколько выходов. Подходит для более подробного результата

import numpy as np

# Матрица весовых коэффициентов
ih_wgt = np.array([
    [0.1, 0.2, -0.1],
    [-0.1, 0.1, 0.9],
    [0.1, 0.4, 0.1]
]).T

hp_wgt = np.array([ # Матрица входных значений
    [0.3, 1.1, -0.3], # поломка
    [0.1, 0.2, 0.0], # ремонт
    [0.0, 1.3, 0.1] # смотр
]).T

weights = [ih_wgt, hp_wgt]

def neural_network(_input, weights):

    hid = _input.dot(weights[0])
    pred = hid.dot(weights[1])
    return pred

breaking = np.array([0, 0, 0, 1, 2, 1, 3, 0, 0, 1, 4]) # Количество поломок оборудования за 11 лет
repair = np.array([0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 3]) # Сколько раз оборудование ремонтировалось
review = np.array([1, 1, 2, 0, 0, 0, 0, 3, 1, 2, 0]) # Сколько раз оборудование было осмотрено

_input = np.array([breaking[10], repair[10], review[10]])

pred = neural_network(_input, weights)

print(pred)