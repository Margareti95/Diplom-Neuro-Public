# Нейронная сеть №2. Вероятность гибели персонала в случае катастрофы
# Тут временные данные по общей статистики по погибшим

import numpy as np

emergency = np.array([0.17, 0.15, 0.14, 0.18, 0.18, 0.16, 0.13, 0.15, 0.14, 0.14, 0.12, 0.09, 0.18, 0.19, 0.15])
doom = np.array([0.16, 0.14, 0.16, 0.16, 0.18, 0.18, 0.22, 0.15, 0.09, 0.13, 0.21, 0.14, 0.08, 0.32, 0.12])
affected = np.array([0.08, 0.10, 0.12, 0.11, 0.09, 0.05, 0.15, 0.11, 0.13, 0.14, 0.09, 0.10, 0.18, 0.19, 0.12])

weights = np.array([0.95, 0.95, 0.96])

_input = np.array([emergency[12], doom[12], affected[12]])

def neural_network(_input, weights):
    pred = _input.dot(weights)
    return pred

pred = neural_network(_input, weights)

print(pred)