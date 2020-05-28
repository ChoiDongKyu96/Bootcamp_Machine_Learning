# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_train.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/27 16:54:43 by dochoi            #+#    #+#              #
#    Updated: 2020/05/28 00:33:02 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
import pandas as pd

def minmax(x):
    pivot = max(x) - min(x)
    return (x - min(x)) / pivot

def zscore(x):
    if len(x) == 0:
        return none
    mu = sum(x)/len(x)
    temp = 0.0
    for elem in x:
        temp += ((elem - mu) * (elem - mu))
    var = temp
    std = math.sqrt(var/ (len(x) - 1))
    return (x - mu) / std

csv_data = pd.read_csv("../resources/are_blue_pills_magics.csv")
y_n = []
x = np.array(csv_data["Micrograms"]).reshape(-1,1)
x = minmax(x)

y = np.array(csv_data["Score"]).reshape(-1,1)
y = minmax(y)
plt.scatter(x,y)
x9 = add_polynomial_features(x,9)
# mylr4 = MyLR([[10.0],[-21.0 ], [-0.28], [4.63], [6.73]],alpha=5e-3)
mylr9 = MyLR([[  0.99549772],
 [ -3.04228406],
 [ 11.0342294 ],
 [-12.5192794 ],
 [ -7.56251887],
 [  4.59267205],
 [  9.57475922],
 [  5.99224473],
 [ -1.55560663],
 [ -7.52630899]],alpha=0.55)
mylr9.fit_(x9, y)
print(mylr9.cost_(x9, y))

continuous_x = np.arange(0,1, 0.001).reshape(-1,1)
x_9 = add_polynomial_features(continuous_x, 9)
y_hat = mylr9.predict_(x_9)
print(mylr9.thetas)
# print(x_9)
# print(y)
plt.plot(continuous_x, y_hat, color='orange')
plt.show()

