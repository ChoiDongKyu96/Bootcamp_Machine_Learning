# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    test.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 17:17:31 by dochoi            #+#    #+#              #
#    Updated: 2020/05/03 20:56:46 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from matplotlib import pyplot as plt

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
     Args:
     x: has to be an numpy.ndarray, a vector of dimension m * 1.
     y: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
     Returns: Nothing.
     Raises: This function should not raise any Exceptions. """
    def predict_(x, theta):
        def add_intercept(x):
            if len(x) == 0 or x.ndim >= 3:
                return None
            if x.ndim == 1:
                return np.vstack((np.ones(len(x)), x)).T
            else:
                return np.insert(x, 0, 1, axis=1)
        if len(theta) != 2 or len(x) == 0:
            return None
        return add_intercept(x) @ theta


    def cost_(y, y_hat):
        if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
            return None
        return ((y_hat - y) @ (y_hat - y)) / (2 * len(y))

    plt.scatter(theta[1], cost_(y,predict_(x, theta)), color='orange')
    plt.xlabel('theta1')
    plt.ylabel('j()')
    plt.title('simple plot')
    temp = []
    temp = np.arange(-3 , 3, 0.1)
    temp2 = []
    for i in temp :
        theta[1] = i
        temp2.append(cost_(y,predict_(x, theta)))
    plt.plot(temp, temp2)
    # plt.plot(x, predict_(x, theta), color='orange')
    plt.show()

import numpy as np

x = np.arange(1, 6)
y =  np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
theta1 = np.array([4.5, -0.2])
theta2= np.array([-1.5,2])
plot(x, y, theta1)
plot(x, y, theta2)
theta3 = np.array([3, 0.3])
plot(x, y, theta3)