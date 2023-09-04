# Нейронка №3. Аварии

import numpy as np

ih_wgt = np.array([
    [0.1, 0.2, -0.1], # Взрыв
    [-0.1, 0.1, 0.9], # Пожар
    [0.1, 0.4, 0.1] # Выброс опасных веществ
]).T

hp_wgt = np.array([ # Матрица входных значений
    [0.3, 1.1, -0.3], # Взрыв
    [0.1, 0.2, 0.0], # Пожар
    [0.0, 1.3, 0.1] # Выброс опасных веществ
]).T

explosion = np.array([6, 8, 6, 2, 3])
ejection = np.array([11, 3, 9, 9, 12])
fire = np.array([2, 7, 4, 1, 3])

_input = np.array([explosion[4], ejection[4], fire[4]])

weights = [ih_wgt, hp_wgt]

def neural_network(_input, weights):
    hid = _input.dot(weights[0])
    pred = hid.dot(weights[1])
    return pred

pred = neural_network(_input, weights)

print("Прогноз:" + " ", pred)