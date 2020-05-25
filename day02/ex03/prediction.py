# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/25 17:45:53 by dochoi            #+#    #+#              #
#    Updated: 2020/05/25 17:45:54 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 16:44:09 by dochoi            #+#    #+#              #
#    Updated: 2020/05/01 22:24:28 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
     Args:
     x: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
     Returns: y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray. None if x or theta dimensions are not appropriate.
    Raises: This function should not raise any Exceptions. """
    def add_intercept(x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)

    if x.ndim == 1:
        x = x[:,np.newaxis]
    if len(theta) - 1 != x.shape[1]  or len(x) == 0:
        return None
    return add_intercept(x) @ theta

# import numpy as np

# x = np.arange(1, 6, dtype=float)
# theta1 = np.array([5,0])
# print(predict_(x, theta1))
# theta2 = np.array([0,1])
# print(predict_(x, theta2))
# theta3 = np.array([5,3])
# print(predict_(x, theta3))
# theta4 = np.array([-3,1])
# print(predict_(x, theta4))
