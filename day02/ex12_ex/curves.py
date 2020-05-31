# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    curves.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/27 16:35:55 by dochoi            #+#    #+#              #
#    Updated: 2020/05/31 18:12:16 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

def add_polynomial_features(x, power):
    temp = x.copy()
    for i in range(2, power + 1):
        temp = np.append(temp, np.power(x, i), axis=1 )
    return temp

x = np.arange(1,11).reshape(-1,1)
y = np.array([[ 1.39270298], [ 3.88237651], [ 4.37726357], [ 4.63389049], [ 7.79814439], [ 6.41717461], [ 8.63429886], [ 8.19939795], [10.37567392], [10.68238222]])
plt.scatter(x,y)
plt.show()

from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
# Build the model:
x_ = add_polynomial_features(x, 3)
my_lr = MyLR(np.ones(4).reshape(-1,1))
my_lr.fit_(x_, y)
## To get a smooth curve, we need a lot of data points
continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
x_ = add_polynomial_features(continuous_x, 3)
y_hat = my_lr.predict_(x_)
plt.scatter(x,y)
# print(my_lr.thetas)
plt.plot(continuous_x, y_hat, color='orange')
plt.show()
