"""
    Code by Tae-Hwan Hung(@graykode)
    https://en.wikipedia.org/wiki/Dirichlet_distribution
    3-Class Example
"""
from decimal import Decimal
from random import randint

import numpy as np
from matplotlib import pyplot as plt


class dirichlet:

    def __init__(self, dimension):
        self.dimentsion = dimension
        
    def generate_x(self, num_each_dimension=1000):
        x=[]
        axis = np.arange(0, 1, 1/num_each_dimension, dtype=np.float64)
        for i in range(len(axis)):
            for j in range(len(axis)):
                if axis[i]+axis[j]<1 and axis[i] != 0 and axis[j] !=0: #0**0 make error!
                    x.append([axis[i],axis[j],1-axis[i]-axis[j]])
        
        return x

    def gamma_function(self, n):
        cal = 1
        for i in range(2, n):
            cal *= i
        return cal

    def beta_function(self, alpha):
        """
        :param alpha: list, len(alpha) is k
        :return:
        """
        numerator = 1
        for a in alpha:
            numerator *= self.gamma_function(a)
        denominator = self.gamma_function(sum(alpha))

        return Decimal(numerator) / Decimal(denominator)

    def dirichlet(self, x, a):
        """
        :param x: list of [x[1,...,K], x[1,...,K], ...], shape is (n_trial, K)
        :param a: list of coefficient, a_i > 0
        :return:
        """
        c = (1 / self.beta_function(a))
        # print("c",c)
        
        y = []
        for xn in x:
            yn = c
            for i in range(self.dimentsion):
                yn *= Decimal.from_float(xn[i]) ** (a[i] - 1)
            y.append(yn)
            
        # y = [c*(Decimal.from_float(xn[0]) ** (a[0] - 1)) * (Decimal.from_float(xn[1]) ** (a[1] - 1))
        #     * (Decimal.from_float(xn[2]) ** (a[2] - 1)) for xn in x]
        
        return x, y, np.mean(y), np.std(y)

    def sample_dirichlet(self, x, y, sample_num):   
        sample_x = []
        x_index_list = np.random.choice(len(x), sample_num, p=y)
        for index in x_index_list:
            sample_x.append(x[index])
        return sample_x


    def marginal_beta(self, i, alpha):
        alpha0 = sum(alpha)
        alpha_i = alpha[i]
        
        # beta distribution
        x = np.arange(0, 1, 0.001, dtype=np.float64)
        a = alpha_i
        b = alpha0 - alpha_i
        gamma = self.gamma_function(a + b) / \
                (self.gamma_function(a) * self.gamma_function(b))
        y = gamma * (x ** (a - 1)) * ((1 - x) ** (b - 1))
        return x, y, np.mean(y), np.std(y)
        


