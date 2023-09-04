# Усложненный вариант. Несколько входов - несколько выходов. Подходит для более подробного результата
# Используем его

import numpy as np

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

breaking = np.array([0, 0, 0, 1, 2, 1, 3, 0, 0, 1, 4]) # Количество поломок оборудования за 11 лет
repair = np.array([0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 3]) # Сколько раз оборудование ремонтировалось
review = np.array([1, 1, 2, 0, 0, 0, 0, 3, 1, 2, 0]) # Сколько раз оборудование было осмотрено

_input = np.array([breaking[10], repair[10], review[10]]) # Здесь будут содержаться выгрузка в ".csv"
weights = [ih_wgt, hp_wgt]

def neural_network(_input, weights):
    hid = _input.dot(weights[0])
    pred = hid.dot(weights[1])
    return pred

pred = neural_network(_input, weights)

if pred[0] < 0:
    pred[0] = 0

elif pred[1] < 0:
    pred[1] = 0

elif pred[2] < 0:
    pred[2] = 0

print("Прогноз:" + " ", pred)