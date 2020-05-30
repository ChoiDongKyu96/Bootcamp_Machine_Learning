# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_reg_model.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 02:56:12 by dochoi            #+#    #+#              #
#    Updated: 2020/05/30 18:03:49 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter

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

csv_data_x = pd.read_csv("../resources/solar_system_census.csv")

csv_data_y = pd.read_csv("../resources/solar_system_census_planets.csv")

x = np.array(csv_data_x[["height","weight","bone_density"]])
y =  np.array(csv_data_y["Origin"]).reshape(-1,1)

x = zscore(x)


temp = data_spliter(x, y, 0.5)
x_train = temp[0]
x_test = temp[1]
y_train = temp[2]
y_test = temp[3]

y_train0 = np.array([1 if i == 0 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
y_test0 = np.array([1 if i == 0 else 0 for i in y_test]).reshape(-1,1)

y_train1 = np.array([1 if i == 1 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
y_test1 = np.array([1 if i == 1 else 0 for i in y_test]).reshape(-1,1)

y_train2 = np.array([1 if i == 2 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
y_test2 = np.array([1 if i == 2 else 0 for i in y_test]).reshape(-1,1)

y_train3 = np.array([1 if i == 3 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
y_test3 = np.array([1 if i == 3 else 0 for i in y_test]).reshape(-1,1)

mylr0 = MyLR([[-1.32069828],
 [-1.02177506],
 [-0.64913889],
 [-0.06329356]]) # The ﬂying cities of Venus (0)
mylr0.fit_(x_train, y_train0)
mylr0.alpha = 0.03
mylr0.fit_(x_train, y_train0)
mylr0.alpha = 0.3
mylr0.fit_(x_train, y_train0)

mylr1 = MyLR([[-1.56373886],
 [-0.58824757],
 [ 0.28303058],
 [ 2.20809316]]) #  United Nations of Earth (1)
mylr1.fit_(x_train, y_train1)
mylr1.alpha = 0.03
mylr1.fit_(x_train, y_train1)
mylr1.alpha = 0.3
mylr1.fit_(x_train, y_train1)

mylr2 = MyLR([[-2.58616195],
 [ 0.60780971],
 [ 2.8277886 ],
 [ 0.32890994]]) # Mars Republic (2)
mylr2.fit_(x_train, y_train2)
mylr2.fit_(x_train, y_train2)
mylr2.alpha = 0.03
mylr2.fit_(x_train, y_train2)
mylr2.alpha = 0.3
mylr2.fit_(x_train, y_train2)

mylr3 = MyLR([[-4.41035678],
 [ 4.24667587],
 [-3.76787019],
 [-5.23183696]]) # The Asteroids’ Belt colonies (3).
mylr3.fit_(x_train, y_train0)
mylr3.alpha = 0.03
mylr3.fit_(x_train, y_train0)
mylr3.alpha = 0.3
mylr3.fit_(x_train, y_train0)
mylr3.fit_(x_train, y_train3)

print(mylr0.thetas)
print(mylr1.thetas)
print(mylr2.thetas)
print(mylr3.thetas)
# 모델 생성 완료
# 전체 데이터 예측
y_hat0 = mylr0.predict_(x)
y_hat1 = mylr1.predict_(x)
y_hat2 = mylr2.predict_(x)
y_hat3 = mylr3.predict_(x)


y_hat_total = np.append(y_hat0, y_hat1, axis=1)
y_hat_total = np.append(y_hat_total, y_hat2, axis=1)
y_hat_total = np.append(y_hat_total, y_hat3, axis=1)

y_hat_pre_all = np.array([])
# 데이터 확률 최댓값을 기준으로 클래스 분류
for i in range(len(y_hat_total)):
    y_hat_pre_all = np.append(y_hat_pre_all, np.argmax(y_hat_total[i]))

y_hat_pre_all = y_hat_pre_all.reshape(-1,1)
# 시각화
y_n = np.array([0.,0.,0.,0.])
for i in range(len(y)):
    if y[i] == y_hat_pre_all[i]:
        if y[i] == 0:
            y_n[0] += 1
        elif y[i] == 1:
            y_n[1] += 1
        elif y[i] == 2:
            y_n[2] += 1
        elif y[i] == 3:
            y_n[3] += 1

y_n[0] /= np.count_nonzero(y_hat_pre_all == 0)
y_n[1] /= np.count_nonzero(y_hat_pre_all == 1)
y_n[2] /= np.count_nonzero(y_hat_pre_all == 2)
y_n[3] /= np.count_nonzero(y_hat_pre_all == 3)
plt.bar(range(0, 4), y_n * 100,color=['black', 'red', 'green', 'blue'])

plt.xlabel('class(0,1,2,3)')
plt.xticks(range(0, 4))
plt.ylabel('percentage')
plt.title('**Accuarcy**')
plt.show()
