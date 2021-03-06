# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    test.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/29 01:17:02 by dochoi            #+#    #+#              #
#    Updated: 2020/05/29 01:17:44 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

y = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
ones = np.ones(y.shape[0	]).reshape((-1,1))
print(ones - y)

y = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
print(1 - y)