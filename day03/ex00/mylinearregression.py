# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mylinearregression.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/27 16:39:10 by dochoi            #+#    #+#              #
#    Updated: 2020/05/28 17:25:29 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

class MyLinearRegression(object):
    """ Description: My personnal linear regression class to fit like a boss. """
    def __init__(self, thetas, alpha=0.000001, n_cycle=30000):
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
            self.thetas -= self.alpha * ((self.gradient(x, y)))
            # for i, v in enumerate(self.gradient(x, y)):
            #     self.thetas[i] -= (self.alpha * v)
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