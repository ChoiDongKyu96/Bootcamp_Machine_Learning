# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_logistic_regression.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 02:00:53 by dochoi            #+#    #+#              #
#    Updated: 2020/06/01 01:33:40 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math



class MyLogisticRegression(object):
    """ Description: My personnal logistic regression to classify things. """

    def __init__(self, thetas, alpha=0.0001, n_cycle=5000, penalty='l2', lambda_=0.5):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.penalty=penalty
        self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)
        self.lambda_ = lambda_

    def l2(self, theta):
        if len(theta) == 0:
            return None
        l2_lst = theta * theta
        return np.sum(l2_lst) - l2_lst[0]

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
        if self.penalty == 'l2':
            answer = self.add_intercept(x).T @ (self.predict_(x) - y) + (self.lambda_ * self.thetas)
            answer[0] -= self.lambda_ * self.thetas[0]
            return answer / len(x)
        else:
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
        if (len(y) != len(x) or len(y) == 0 or len(x) == 0):
            return None
        y_hat = self.predict_(x) - eps
        if self.penalty == 'l2':
            return -sum((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)) ).squeeze() / len(y_hat) + (self.lambda_ * self.l2(self.thetas)/(2 * len(y_hat)))
        else :
            return -sum((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)) ).squeeze() / len(y_hat)
