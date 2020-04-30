# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 16:01:27 by dochoi            #+#    #+#              #
#    Updated: 2020/04/30 16:40:16 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# import numpy as np

def simpl_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
     Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
      Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray. None if x or theta dimensions are not appropriate.
     Raises: This function should not raise any Exception.
    """
    if len(theta) != 2 or len(x) == 0:
        return None
    return theta[0] + (theta[1] * x[::])

# x = np.arange(1, 6, dtype=float)
# theta1 = np.array([5,0])
# print(simpl_predict(x, theta1))
# theta2 = np.array([0,1])
# print(simpl_predict(x, theta2))
# theta3 = np.array([5,3])
# print(simpl_predict(x, theta3))
# theta4 = np.array([-3,1])
# print(simpl_predict(x, theta4))
