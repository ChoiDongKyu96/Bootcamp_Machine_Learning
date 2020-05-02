# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/01 22:26:04 by dochoi            #+#    #+#              #
#    Updated: 2020/05/02 18:51:06 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from matplotlib import pyplot as plt
import numpy as np

def plot_with_cost(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
      y: has to be an numpy.ndarray, a vector of dimension m * 1.
      theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception. """
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

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')


    y_hat = predict_(x, theta)
    title_str = "Cost: %f" % cost_(y, y_hat)
    plt.title(title_str)
    plt.plot(x, y_hat, color='orange')
    for i in range(len(x)):
        plt.plot((x[i], x[i]), (y[i], y_hat[i]), '--',color='red')
    plt.show()

x = np.arange(1, 6)
y =  np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
theta1 = np.array([4.5, -0.2])
theta2= np.array([-1.5,2])
plot_with_cost(x, y, theta1)
plot_with_cost(x, y, theta2)
theta3 = np.array([3, 0.3])
plot_with_cost(x, y, theta3)
