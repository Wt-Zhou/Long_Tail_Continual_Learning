"""
    Code by Tae-Hwan Hung(@graykode)
    https://en.wikipedia.org/wiki/Dirichlet_distribution
    3-Class Example
"""
from decimal import Decimal
from random import randint

import numpy as np
from matplotlib import pyplot as plt


def normalization(x, s):
    """
    :return: normalizated list, where sum(x) == s
    """
    return [(i * s) / sum(x) for i in x]

def sampling():
    return normalization([randint(1, 100),
            randint(1, 100), randint(1, 100)], s=1)
            # randint(1, 100)], s=1)
    
def generate_x(dimension, num_each_dimension):
    x=[]
    axis = np.arange(0, 1, 1/num_each_dimension, dtype=np.float64)
    for i in range(len(axis)):
        for j in range(len(axis)):
            if axis[i]+axis[j]<1 and axis[i] != 0 and axis[j] !=0: #0**0 make error!
                x.append([axis[i],axis[j],1-axis[i]-axis[j]])
     
    # np.linspace([0,0],[1,1],100,endpoint=True,retstep=False,dtype=float)
    return x



def gamma_function(n):
    cal = 1
    for i in range(2, n):
        cal *= i
    return cal

def beta_function(alpha):
    """
    :param alpha: list, len(alpha) is k
    :return:
    """
    numerator = 1
    for a in alpha:
        numerator *= gamma_function(a)
    denominator = gamma_function(sum(alpha))

    return Decimal(numerator) / Decimal(denominator)

def dirichlet(x, a):
    """
    :param x: list of [x[1,...,K], x[1,...,K], ...], shape is (n_trial, K)
    :param a: list of coefficient, a_i > 0
    :return:
    """
    c = (1 / beta_function(a))
    print("c",c)
        
    y = [c*(Decimal.from_float(xn[0]) ** (a[0] - 1)) * (Decimal.from_float(xn[1]) ** (a[1] - 1))
         * (Decimal.from_float(xn[2]) ** (a[2] - 1)) for xn in x]

    
    return x, y, np.mean(y), np.std(y)

def sample_dirichlet(x, y, sample_num):   
    sample_x = []
    x_index_list = np.random.choice(len(x), sample_num, p=y)
    for index in x_index_list:
        sample_x.append(x[index])
    print("sample_x",sample_x)
    return 


def marginal_beta(i, alpha):
    alpha0 = sum(alpha)
    alpha_i = alpha[i]
    
    # beta distribution
    x = np.arange(0, 1, 0.001, dtype=np.float64)
    a = alpha_i
    b = alpha0 - alpha_i
    gamma = gamma_function(a + b) / \
            (gamma_function(a) * gamma_function(b))
    y = gamma * (x ** (a - 1)) * ((1 - x) ** (b - 1))
    return x, y, np.mean(y), np.std(y)
    



for ls in [(1, 1, 1),(200, 300, 300),(1, 1, 10),(1, 1, 15)]:

    alpha = list(ls)

    # marginal_beta
    # for i in range(3):
    #     x, y, u, s = marginal_beta(i, alpha)
    #     plt.plot(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f,'
    #                          r'\ \alpha=%d,\ \beta=%d$' % (u, s, alpha[i], sum(alpha)-alpha[i]))
    
    # random samping [x[1,...,K], x[1,...,K], ...], shape is (n_trial, K)
    # each sum of row should be one.
    # x = [sampling() for _ in range(1, n_experiment + 1)]
    x = generate_x(3, 1000)
    x, y, u, s = dirichlet(x, alpha)
    sumy = sum(y)
    print("y",sumy/len(x))
    
    nor_y = [item / (sumy) for item in y]
    sample_dirichlet(x, nor_y, sample_num=10)
    
    plt.plot(x, y, label=r'$\alpha=(%d,%d,%d)$' % (ls[0], ls[1], ls[2]))

    
    
plt.legend()
plt.savefig('graph/dirichlet.png')
plt.show()
