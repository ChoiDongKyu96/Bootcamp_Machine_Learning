# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_reg_model.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 02:56:12 by dochoi            #+#    #+#              #
#    Updated: 2020/05/30 02:57:59 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as MyLR

csv_data = pd.read_csv("../resources/are_blue_pills_magics.csv")
y_n = []
x = np.array(csv_data["Micrograms"]).reshape(-1,1)
x = minmax(x)

y = np.array(csv_data["Score"]).reshape(-1,1)
y = minmax(y)