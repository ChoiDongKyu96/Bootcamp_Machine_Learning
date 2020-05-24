# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_model.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/24 12:04:43 by dochoi            #+#    #+#              #
#    Updated: 2020/05/24 19:05:45 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from matplotlib import pyplot as plt
import numpy as np


import pandas as pd
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR



data = pd.read_csv("../../resources/are_blue_pills_magics.csv")

Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
# linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
# Y_model2 = linear_model2.predict_(Xpill)

print(linear_model1.cost_(Xpill, Yscore)) # 57.60304285714282 >>>
print(mean_squared_error(Yscore, Y_model1)) # 57.603042857142825 >>>
# print(linear_model2.cost_(Xpill, Yscore)) # 232.16344285714285
# print(mean_squared_error(Yscore, Y_model2))

x= Xpill
y = Yscore
plt.scatter(x, y)
linear_model1.fit_(x, y)
plt.xlabel('Quantity of blue pill (in micrograms)')
plt.ylabel('Space driving score')
plt.title('simple plot')
plt.plot(x, linear_model1.predict_(x), color='green')
plt.legend(['S_true', 'S_predict'])
plt.show()




legends = []
plt.xlabel('theta1')
plt.ylabel('cost func J(theta0, theta1)')
theta0s = np.arange(linear_model1.thetas[0] - 15, linear_model1.thetas[0] + 15, 5)
theta1 = linear_model1.thetas[1].copy()
print(linear_model1.thetas)
for theta0 in theta0s:
    linear_model1.thetas[0]= theta0
    linear_model1.thetas[1] = theta1
    legends.append('theta0 = ' + str(theta0))
    temp = np.arange(theta1 - 5 , theta1 + 5, 0.1)
    # print(linear_model1.thetas)
    temp2 = []
    for i in temp :
        linear_model1.thetas[1] = i
        temp2.append(linear_model1.cost_(x,y))
    plt.plot(temp, temp2)
    print(min(temp2))
plt.legend(legends)
plt.ylim([20, 140])
plt.show()