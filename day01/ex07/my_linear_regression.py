# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/23 16:09:10 by dochoi            #+#    #+#              #
#    Updated: 2020/05/24 18:57:08 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class MyLinearRegression(object):
    """ Description: My personnal linear regression class to fit like a boss. """
    def __init__(self, thetas, alpha=0.001, n_cyle=1000):
        self.alpha = alpha
        self.max_iter = n_cyle
        self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)

    def add_intercept(self, x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)
    def gradient(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        if x.shape != y.shape:
            return None
        return (self.add_intercept(x).T @ (self.predict_(x).reshape(-1,1) - y)) / len(x)
    def fit_(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        if x.shape != y.shape:
            return None
        max_iter = self.max_iter
        while max_iter:
            for i, v in enumerate(self.gradient(x, y)):
                self.thetas[i] -= (self.alpha * v)
            max_iter -= 1
        return self.thetas

    def predict_(self,x):
        if len(self.thetas) != 2 or len(x) == 0:
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
        return (sum((y_hat - y) * (y_hat - y)).squeeze()) / (len(y))

# x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
# y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
# lr1 = MyLinearRegression([2, 0.7])


# print(lr1.predict_(x))
# # # Output: array([[10.74695094], [17.05055804], [24.08691674], [36.24020866], [42.25621131]])
# # # Example 0.1:
# print(lr1.cost_elem_(lr1.predict_(x),y))
# #  # Output: array([[77.72116511], [49.33699664], [72.38621816], [37.29223426], [78.28360514]])
# # # Example 0.2:
# print(lr1.cost_(lr1.predict_(x),y))
# # # Output: 315.0202193084312
# # # Example 1.0:
# lr2 = MyLinearRegression([0, 0])

# lr2.fit_(x, y)
# print(lr2.thetas)
#  # Output: array([[1.40709365], [1.1150909 ]])
# # # Example 1.1:
# print(lr2.predict_(x))
# #  # Output: array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])
# # # Example 1.2:
# print(lr2.cost_elem_(lr1.predict_(x),y)) # Output: array([[35.6749755 ], [ 4.14286023], [ 1.26440585], [29.30443042], [22.27765992]])
# # # Example 1.3:
# print(lr2.cost_(lr1.predict_(x),y))
# # # Output: 92.66433192085971
