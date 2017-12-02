# -*- coding: utf-8 -*-
"""
Created on 2017/12/02

Using Numpy, Pandas and Matplotlib to emplot Linear Regression by Gradient Descent

@author: HeHeHe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

class DimensionValueError(ValueError):
    """Define Abnoraml Class"""
    pass

class MyLinearRegression(object):
    """Define Linear Regression Class"""
    def __init__(self, X, y, theta, alpha, num_iters):
        """X should inclue intercept column"""
        self.X = X
        slef.y = y

        self.dimension = X.shape[1]

        if theta.shape[0] != self.dimension:
            raise DimensionValueError("N variables error")

        self.theta = theta
        self.alpha = alpha
        self.num_iters = num_iters
        self.m = y.size


def compute_cost(self):
    """
    Compute Loss function for Linear Regression
    """
    
    cost= (1.0/(2*self.m))*sum((self.X.dot(self.theta)- self.y).flatten()**2)

    return cost

def gradient_descent(self):
    """Perform gradient descent to learn theta
    by taking num_items gradient steps with 
    learning rate alpha
    """

    Cost = []

    for i in range(self.num_iters):
        tempTheta = self.theta

        tempTheta[0][0] = self.theta[0][0] - self.alpha*(1.0/m)*sum((self.X.dot(self.theta).flatten() - self.y)*self.X[:, 0])
        tempTheta[1][0] = self.theta[1][0] - self.alpha*(1.0/m)*sum((self.X.dot(self.theta).flatten() - self.y)*self.X[:, 1])

        self.theta[0][0] = tempTheta[0]
        self.theta[1][0] = tempTheta[1]

        Cost.append(self.compute_cost())
    return self.theta, Cost

data = np.array([
        [1. , 1.1],
        [0.9, 0.98],
        [1.01, 1.02],
        [2., 2.],
        [2.1, 1.98],
        [1.89, 1.99],
        [3., 2.98],
        [3.01, 3],
        [2.79, 2.88],
        [4. , 4.1],
        [4.01, 4.03],
        [3.99, 4.]
        ])
x = data[:, 0]
y = data[:, 1]
m = y.size
X = np.ones(shape = (m,2))
X[:, 1] = x
theta = np.zeros(shape = (2,1))
num_iters = 1000
alpha = 0.01

if __name__ =='__main__':
    LR = MyLinearRegression(X, y, theta, alpha, num_iters)
    a, b = LR.gradient_descent()
