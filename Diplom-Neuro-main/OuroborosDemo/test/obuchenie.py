# Снизу нейронное обучение, а не прогнозирование
import numpy as np
goal_pred = 0.8
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

breaking = np.array([0, 0, 0, 1, 2, 1, 3, 0, 0, 1, 4]) # Количество поломок оборудования за 11 лет
repair = np.array([0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 3]) # Сколько раз оборудование ремонтировалось
review = np.array([1, 1, 2, 0, 0, 0, 0, 3, 1, 2, 0]) # Сколько раз оборудование было осмотрено

_input = np.array([breaking[10], repair[10], review[10]])

weights = [ih_wgt, hp_wgt]

for i in range(70):
    hid = _input.dot(weights[0])
    pred = hid.dot(weights[1])
    error = (pred - goal_pred) ** 2 # Вычисление чистой ошибки
    gradient = (pred - goal_pred) * _input # Градиентный спуск
    weights[1] -= gradient

    #print("Ошибка:" + " " + str(error), " " + "Прогноз:" + " " + str(pred))
    print("Прогноз:" + " " + str(pred))