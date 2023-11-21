import math
import numpy as np
from scipy.optimize import minimize


# aim func: cross entropy
def func(x):
    fun = 0
    for i in range(len(x)):
        fun += x[i] * math.log10(x[i])
    return fun


# constraint 1: the sum of weights is 1
def cons1(x):
    return sum(x)


# constraint 2: define tolerance to imprecision
def cons2(x):
    tol = 0
    for i in range(len(x)):
        tol += (len(x) - (i+1)) * x[i] / (len(x) - 1)
    return tol


# function for power set
def PowerSetsBinary(items):
    # generate all combination of N items
    N = len(items)
    # enumerate the 2**N possible combinations
    set_all=[]
    for i in range(2**N):
        combo = []
        for j in range(N):
            if(i >> j) % 2 == 1:
                combo.append(items[j])
        set_all.append(combo)
    return set_all


def class_set_utility_matrix(num_class=4):
    # compute the weights g for ordered weighted average aggreagtion
    for j in range(2, (num_class + 1)):
        num_weights = j
        ini_weights = np.asarray(np.random.rand(num_weights))

        name = 'weight' + str(j)
        locals()['weight' + str(j)] = np.zeros([5, j])

        for i in range(5):
            tol = 0.5 + i * 0.1

            cons = ({'type': 'eq', 'fun': lambda x: cons1(x) - 1},
                    {'type': 'eq', 'fun': lambda x: cons2(x) - tol},
                    {'type': 'ineq', 'fun': lambda x: x - 0.00000001}
                    )

            res = minimize(func, ini_weights, method='SLSQP', options={'disp': True}, constraints=cons)
            locals()['weight' + str(j)][i] = res.x
            # print(res.x)

    class_set = list(range(num_class))
    act_set = PowerSetsBinary(class_set)
    act_set.remove(act_set[0])  # emptyset is not needed
    act_set = sorted(act_set)
    # print('act_set:', act_set)
    # print('length of act set:', len(act_set))
    # label_dict = {0:'Normal', 1:'Pneumonia-Bacterial', 2:'Pneumonia-Bacterial', 3:'COVID-19'}

    utility_matrix = np.zeros([len(act_set), len(class_set)])
    tol_i = 1
    ''''''
    # tol_i = 0 with tol=0.5, tol_i = 1 with tol=0.6, tol_i = 2 with tol=0.7, tol_i = 3 with tol=0.8, tol_i = 4 with tol=0.9
    ''''''
    for i in range(len(act_set)):
        intersec = class_set and act_set[i]
        if len(intersec) == 1:
            utility_matrix[i, intersec] = 1

        else:
            for j in range(len(intersec)):
                utility_matrix[i, intersec[j]] = locals()['weight' + str(len(intersec))][tol_i, 0]
    # print(utility_matrix)
    return act_set, utility_matrix












