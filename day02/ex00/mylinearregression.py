# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mylinearregression.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/25 16:42:45 by dochoi            #+#    #+#              #
#    Updated: 2020/05/25 16:46:49 by dochoi           ###   ########.fr        #
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
