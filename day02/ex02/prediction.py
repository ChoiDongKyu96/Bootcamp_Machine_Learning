# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/25 17:32:42 by dochoi            #+#    #+#              #
#    Updated: 2020/05/25 17:51:34 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
     Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
      Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray. None if x or theta dimensions are not appropriate.
     Raises: This function should not raise any Exception.
    """
    if len(x) == 0:
        return None
    answer = []
    for i in range(len(x)):
        temp = theta[0]
        for j in range(len(x[0])):
            temp += (x[i][j] * theta[j + 1])
        answer.append(temp)
    return (answer)


x = np.arange(1,13).reshape((4,-1))
theta1 = np.array([5, 0, 0, 0])
print(simple_predict(x, theta1))

theta2 = np.array([0, 1, 0, 0])
print(simple_predict(x, theta2)) # Output: array([ 1., 4., 7., 10.])

theta4 = np.array([-3, 1, 2, 3.5])

print(simple_predict(x,theta4))