# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ridge.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/31 16:02:01 by dochoi            #+#    #+#              #
#    Updated: 2020/05/31 18:47:28 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

def zscore(x):
    if len(x) == 0:
        return none
    mu = sum(x)/len(x)
    temp = 0.0
    for elem in x:
        temp += ((elem - mu) * (elem - mu))
    var = temp
    std = np.sqrt(var/ (len(x) - 1))
    return (x - mu) / std

class MyLinearRegression(object):
    """ Description: My personnal linear regression class to fit like a boss. """
    def __init__(self, thetas, alpha=0.01, n_cycle=300000):
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

class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, n_cycle=30000, lambda_=0.5):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = thetas
        self.lambda_ = lambda_
    def gradient(self, x, y):
        if len(y) == 0 or len(x) == 0 or len(self.thetas) == 0:
            return None
        if len(y) != len(x) or len(x[0]) != len(self.thetas) - 1:
            print(self.thetas)
            return None
        answer = self.add_intercept(x).T @ (self.predict_(x) - y) + (self.lambda_  * self.thetas)
        answer[0] -= self.lambda_  * self.thetas[0]
        return answer / len(x)

    def l2(self, theta):
        if len(theta) == 0:
            return None
        l2_lst = theta * theta
        return np.sum(l2_lst) - l2_lst[0]

    def reg_cost_(self, x, y):
        if (len(y) == 0 or len(x) == 0 or len(self.thetas) == 0):
            return None
        if (len(y) != len(x)):
            return None
        return self.cost_(x, y)  + (self.lambda_ * self.l2(self.thetas)/(2 * len(y)))
