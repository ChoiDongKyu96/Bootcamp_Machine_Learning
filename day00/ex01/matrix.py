# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    matrix.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/29 20:02:32 by dochoi            #+#    #+#              #
#    Updated: 2020/04/30 14:55:35 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from vector import Vector

class Matrix(object):
    def __init__(self, values):
        try :
            if isinstance(values, int):
                self.values = list(range(values))
            elif isinstance(values, list):
                self.values = list(values)
            elif isinstance(values,tuple) and len(values) == 2:
                self.values = list(range(values[0], values[1]))
            self.shape = (len(self.values), len(self.values[0]))
            # for i in range(self.shape):
            #     self.values[i] = float(self.values[i])
        except TypeError:
             print("The values must be able to convert to a float")

    def __add__(self, values2):
        try :
            if isinstance(values2, (int, float)):
                return Matrix(list(list(b+ float(values2) for b in a) for a in self.values)) # list comprehension
            elif not isinstance(values2, Matrix):
                raise ValueError
            if self.shape != values2.shape:
                raise ValueError
            else:
                return Matrix(list(list(c + d for c,d in zip(a , b))for a,b in zip(self.values, values2.values))) # list comprehension
        except TypeError:
            print("The values must be able to convert to a float")
        except ValueError:
            print("They are different sizes.")

    def __radd__(self, values2):
        return (self + values2)

    def __sub__(self, values2):
        try :
            if isinstance(values2, (int, float)):
                return Matrix(list(list(b - float(values2) for b in a) for a in self.values)) # list comprehension
            elif not isinstance(values2, Matrix):
                raise ValueError
            if self.shape != values2.shape:
                raise ValueError
            else:
                return Matrix(list(list(c - d for c,d in zip(a , b))for a,b in zip(self.values, values2.values))) # list comprehension
        except TypeError:
            print("The values must be able to convert to a float")
        except ValueError:
            print("They are different sizes.")

    def __rsub__(self, values2):
        try :
            if isinstance(values2, (int, float)):
                return Matrix(list(list(float(values2) - b for b in a) for a in self.values)) # list comprehension
            elif not isinstance(values2, Matrix):
                raise ValueError
            if self.shape != values2.shape:
                raise ValueError
            else:
                return Matrix(list(list(d - c for c,d in zip(a , b))for a,b in zip(self.values, values2.values))) # list comprehension
        except TypeError:
            print("The values must be able to convert to a float")
        except ValueError:
            print("They are different sizes.")

    def __truediv__(self, values2):
        try :
            if isinstance(values2, (int, float)):
                return Matrix(list(list(b / float(values2) for b in a) for a in self.values)) # list comprehension
            else:
                raise ValueError
        except ValueError:
            print("only scalar")

    def __rtruediv__(self, values2):
        try :
            raise ValueError
        except ValueError:
            print("only scalar")

    def __mul__(self, values2):
        try :
            if isinstance(values2, (int, float)):
                return Matrix(list(list(b * float(values2) for b in a) for a in self.values)) # list comprehension
            elif not isinstance(values2, (Matrix, Vector)):
                raise ValueError
            if isinstance(values2, Vector):
                if self.shape[1] != values2.size:
                    raise ValueError
                answer = []
                for row in self.values:
                    sumidx = 0.0
                    for i in range(len(row)) :
                        sumidx += row[i] * values2.values[i]
                    answer.append([sumidx])
                return answer
            else:
                if self.shape[1] != values2.shape[0]:
                    raise ValueError
                answer = []
                for row in self.values:
                    sublst = []
                    for i in range(values2.shape[1]):
                        sumidx = 0.0
                        for j in range(len(row)) :
                            sumidx += row[j] * values2.values[j][i]
                        sublst.append(sumidx)
                    answer.append(sublst)
                return answer
        except TypeError:
            print("The values must be able to convert to a float")
        except ValueError:
            print("They are different sizes.")

    def __rmul__(self, values2):
        try :
            if isinstance(values2, (int, float)):
                return Matrix(list(list(b * float(values2) for b in a) for a in self.values)) # list comprehension
            elif not isinstance(values2, (Matrix, Vector)):
                raise ValueError
            if isinstance(values2, Vector):
                raise ValueError
            else:
                if self.shape[0] != values2.shape[1]:
                    raise ValueError
                answer = []
                for row in values2.values:
                    sublst = []
                    for i in range(self.shape[1]):
                        sumidx = 0.0
                        for j in range(len(row)) :
                            sumidx += row[j] * self.values[j][i]
                        sublst.append(sumidx)
                    answer.append(sublst)
                return answer
        except TypeError:
            print("The values must be able to convert to a float")
        except ValueError:
            print("They are different sizes.")

    def __repr__(self):
        return '<{0}.{1} object at {2}>'.format(
            self.__module__, type(self).__name__, hex(id(self)))

    def __str__(self):
        return (str(self.values))