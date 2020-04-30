# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    TinyStatistician.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 15:00:15 by dochoi            #+#    #+#              #
#    Updated: 2020/04/30 15:40:16 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import math

class TinyStatistician(object):

    def mean(self, x):
        if len(x) == 0:
            return None
        return sum(x)/len(x)

    def medain(self, x):
        if len(x) == 0:
            return None
        x_sorted_copy = sorted(x)
        return float(x_sorted_copy[int(len(x_sorted_copy) / 2)])

    def quartiles(self, x, percentile):
        if len(x) == 0:
            return None
        x_sorted_copy = sorted(x)
        return float(x_sorted_copy[int(len(x_sorted_copy) * (percentile/ 100))])

    def var(self, x):
        if len(x) == 0:
            return None
        mu = self.mean(x)
        temp = 0.0
        for elem in x:
            temp += ((elem - mu) * (elem - mu))
        return temp / len(x)

    def std(self, x):
        return (math.sqrt(self.var(x)))

tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]
print(tstat.mean(a))
print(tstat.quartiles(a, 25))
print(tstat.quartiles(a, 75))
print(tstat.var(a))
print(tstat.std(a))