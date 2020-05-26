# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multivariate_linear_model.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/27 00:47:48 by dochoi            #+#    #+#              #
#    Updated: 2020/05/27 02:31:55 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class MyLinearRegression(object):
    """ Description: My personnal linear regression class to fit like a boss. """
    def __init__(self, thetas, alpha=5e-5, n_cycle=24000):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)

    def add_intercept(self,x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)

    def gradient(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        return self.add_intercept(x).T @ (self.predict_(x).reshape(-1,1) - y) / len(x)

    def fit_(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        n_cycle= self.n_cycle
        while n_cycle:
            for i, v in enumerate(self.gradient(x, y)):
                self.thetas[i] -= (self.alpha * v)
            n_cycle -= 1
        return self.thetas

    def predict_(self,x):
        if x.ndim == 1:
            x = x[:,np.newaxis]
        if len(self.thetas) - 1 != x.shape[1]  or len(x) == 0:
            return None
        return self.add_intercept(x) @ self.thetas

    def cost_elem_(self, x, y):
        y_hat = self.predict_(x)
        if y.ndim == 1:
            y = y[:,np.newaxis]
        if y.shape != y_hat.shape or len(y) == 0 or len(y_hat) ==0:
            return None
        return ((y - y_hat) ** 2) / (2 * len(y))

    def cost_(self, x, y):
        y_hat = self.predict_(x)
        if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
            return None
        return (sum((y_hat - y) * (y_hat - y)).squeeze() / (2 * len(y)))


csv_data = pd.read_csv("../resources/spacecraft_data.csv")
# x = np.array(csv_data["Age"]).reshape(-1,1)
# y = np.array(csv_data["Sell_price"]).reshape(-1,1)
# plt.scatter(x, y,color="navy", s=10)
# plt.xlabel('x1: age (in years)')
# plt.ylabel('y : Sell_price(in kiloeuros)')
# plt.title('Plot of the selling prices of spacecrafts with respect to their age')
# mylr = MyLinearRegression([[1.], [1.]])
# mylr.fit_(x, y)
# print(mylr.thetas)
# print(mylr.cost_(x, y))
# plt.scatter(x, mylr.predict_(x), color="cornflowerblue", s=7)
# plt.legend(['Sell_price', 'Predicted sell_price'])
# plt.grid()
# plt.show()

# x = np.array(csv_data["Thrust_power"]).reshape(-1,1)
# y = np.array(csv_data["Sell_price"]).reshape(-1,1)

# plt.scatter(x, y,color="forestgreen", s=10)
# plt.xlabel('x1: Thrust_power (in 10km/s)')
# plt.ylabel('y : Sell_price(in kiloeuros)')
# plt.title('Plot of the selling prices of spacecrafts with respect to their thrust')
# mylr = MyLinearRegression([[1.], [1.]])
# mylr.fit_(x, y)
# print(mylr.thetas)
# print(mylr.cost_(x, y))
# plt.scatter(x, mylr.predict_(x), color="greenyellow", s=7)
# plt.legend(['Sell_price', 'Predicted sell_price'])
# plt.grid()
# plt.show()

# x = np.array(csv_data["Terameters"]).reshape(-1,1)
# y = np.array(csv_data["Sell_price"]).reshape(-1,1)
# plt.scatter(x, y,color="purple", s=10)
# plt.xlabel('x1: Terameters (in Tmeters)')
# plt.ylabel('y : Sell_price(in kiloeuros)')
# plt.title('Plot of the selling prices of spacecrafts with respect to their dist')
# mylr = MyLinearRegression([[1000.], [-1.]])
# mylr.fit_(x, y)
# print(mylr.thetas)
# print(mylr.cost_(x, y))
# plt.scatter(x, mylr.predict_(x), color="violet", s=7)
# plt.legend(['Sell_price', 'Predicted sell_price'])
# plt.grid()
# plt.show()

x = np.array(csv_data[['Age','Thrust_power','Terameters']])
x_age = np.array(csv_data["Age"]).reshape(-1,1)
x_thr = np.array(csv_data["Thrust_power"]).reshape(-1,1)
x_dist = np.array(csv_data["Terameters"]).reshape(-1,1)
y = np.array(csv_data["Sell_price"]).reshape(-1,1)
# plt.scatter(x_age, y,color="navy", s=10)
# plt.xlabel('x1: age (in years)')
# plt.ylabel('y : Sell_price(in kiloeuros)')
# plt.title('Plot of the selling prices of spacecrafts with respect to their age(Multivariate )')
mylr = MyLinearRegression([[338.94317973],
 [-22.67763021],
 [  5.84252624],
 [ -2.59281776]])
mylr.fit_(x, y)
# print(mylr.cost_(x, y))
# plt.scatter(x_age, mylr.predict_(x), color="cornflowerblue", s=7)
# plt.legend(['Sell_price', 'Predicted sell_price'])
# plt.grid()
# plt.show()



# plt.scatter(x_thr, y,color="forestgreen", s=10)
# plt.xlabel('x1: Thrust_power (in 10km/s)')
# plt.ylabel('y : Sell_price(in kiloeuros)')
# plt.title('Plot of the selling prices of spacecrafts with respect to their thrust(Multivariate )')
# plt.scatter(x_thr, mylr.predict_(x), color="greenyellow", s=7)
# plt.legend(['Sell_price', 'Predicted sell_price'])
# plt.grid()
# plt.show()

plt.scatter(x_dist, y,color="purple", s=10)
plt.xlabel('x1: Terameters (in Tmeters)')
plt.ylabel('y : Sell_price(in kiloeuros)')
plt.title('Plot of the selling prices of spacecrafts with respect to their dist')
plt.scatter(x_dist, mylr.predict_(x), color="violet", s=7)
plt.legend(['Sell_price', 'Predicted sell_price'])
plt.grid()
plt.show()