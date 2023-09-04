# Пример градиентного спуска

weight = 0.5
goal_pred = 0.8 # Прогноз, который мы хотим получить
_input = 0.5

for iteration in range(120):
    pred = _input * weight
    error = (pred - goal_pred) ** 2
    direction_and_amount = (pred - goal_pred) * _input # градиентный спуск - вычисляем сразу и направление, и величину изменения веса
    weight -= direction_and_amount

    print("Error: " + str(error) + " Prediction: " + str(pred)) # 0.7999999999 - самый близкий к желаемому результату