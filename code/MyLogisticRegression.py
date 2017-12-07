#-*- coding: utf-8 -*-
"""
Created on 2017/12/05

Using Numpy, Pandas and Spicy to build Multi-Class Logistic Regression

@author: HeHeHe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class DimensionValueError(ValueError):
    """Define Abnormal Class"""
    pass

class LogisticRegression(object):
    """Define LogisticRegression Class"""
    def __init__(self):
#        self.X = np.matrix(X)
#        self.y = np.matrix(y)
#        self.theta = np.matrix(theta)
#
#        self.dimension = X.shape[1]
#
#        if theta.shape[0] != self.dimension:
#            raise DimensionValueError("N Variables error")
#
#        self.m = y.size
#        self.learningRate = learningRate
#        self.num_labels = num_labels
         pass

    def sigmoid(self, z):
        return (1.0/(1.0 + np.exp(-1.0*z)))

    def compute_cost(self, theta, X, y, learningRate):
        X = np.matrix(X)
        y = np.matrix(y)
        theta = np.matrix(theta)
        
        first = np.multiply(-y, np.log(self.sigmoid(X*theta.T)))
        second = np.multiply((1-y), np.log(1-self.sigmoid(X*theta.T)))
        reg = (learningRate/2*len(X))*np.sum(np.power(theta[:, 1:theta.shape[1]],2))
        
        return np.sum(first-second)/(len(X)) + reg

    def gradient_descent(self, theta, X, y, learningRate):
        X = np.matrix(X)
        y = np.matrix(y)
        theta = np.matrix(theta)

        parameters = int(theta.flatten().shape[1])
        error = self.sigmoid(X*theta.T) - y

        grad = ((X.T*error)/len(X)).T + ((learningRate/len(X))*theta)
        grad[0,0] = np.sum(np.multiply(error, X[:, 0]))/len(X)

        return np.array(grad).flatten()

    def one_vs_all(self, theta, X, y, learningRate, num_labels):
        X = np.matrix(X)
        y = np.matrix(y)
        theta = np.matrix(theta)

        rows = X.shape[0]
        params = X.shape[1]

        all_theta = np.zeros((num_labels, params))

        for i in range(1, num_labels+1):
            y_i = np.array([1 if label == i else 0 for label in y])
            y_i = np.reshape(y_i, (rows, 1))

            fmin = minimize(fun = self.compute_cost, x0 = theta, args = (X, y_i, learningRate), method = 'TNC', jac = self.gradient_descent)
            all_theta[i-1, :] = fmin.x
        return all_theta

    def predict_all(self, X, all_theta):
        rows = X.shape[0]
        params = X.shape[1]
        num_lables = all_theta.shape[0]

        X = np.matrix(X)
        all_theta = np.matrix(all_theta)

        h = self.sigmoid(X*all_theta.T)
        h_argmax = np.argmax(h, axis = 1)
        h_argmax = h_argmax+1
        return h_argmax

#if __name__ == '__main__':
#    LR = LogisticRegression(X, y, theta)
#    p = LR.fit()
