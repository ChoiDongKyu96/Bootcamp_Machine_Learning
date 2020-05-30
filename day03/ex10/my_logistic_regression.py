# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_logistic_regression.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 02:00:53 by dochoi            #+#    #+#              #
#    Updated: 2020/05/30 17:27:02 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

class MyLogisticRegression(object):
    """ Description: My personnal logistic regression to classify things. """

    def __init__(self, thetas, alpha=0.003, n_cycle=30000):
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

    def log_gradient(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        return np.sum(self.add_intercept(x).T @ (self.predict_(x) - y) / len(x),axis=1).reshape(-1,1)

    def fit_(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        n_cycle= self.n_cycle
        while n_cycle:
            self.thetas -= self.alpha * ((self.log_gradient(x, y)))
            n_cycle -= 1
        return self.thetas

    def predict_(self,x):
        if (len(x) == 0 or len(self.thetas) == 0 or x.shape[1] != (len(self.thetas) - 1)):
            return None
        return 1 / (1 + pow(math.e, -self.add_intercept(x) @ self.thetas))

    def cost_(self, x, y, eps=1e-15):
        y_hat = self.predict_(x) - eps
        if (y.shape != y_hat.shape or len(y) == 0 or len(y_hat) == 0):

            return None
        return -sum((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)) ).squeeze() / len(y_hat)
