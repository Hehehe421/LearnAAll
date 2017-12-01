# -*- coding: utf-8 -*-
"""
Created on 2017/12/01

Using Numpy, Pandas and Matplotlib to employ PCA 

@author: HeHeHe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

class DimensionValueError(ValueError):
    """Define Abnormal Class"""
    pass

class PCA(object):
    """Define PCA Class"""
    def __init__(self, x, n_components = None):
        """x should by ndarray"""
        self.x = x
        self.dimension = x.shape[1]

        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")

        self.n_components = n_components

    def cov(self):
        """Calculate the Covariance Matrix"""
        x_T = np.transpose(self.x)
        x_cov = np.cov(x_T)
        return x_cov

    def get_feature(self):
        """Get EigValues and EigVectors"""
        x_cov = self.cov()
        a, b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m, 1)), b))
        c_df = pd.DataFrame(c)
        c_df_sort = c_df.sort_values([0], ascending = False)
        return c_df_sort

    def explained_variance_(self):
        c_df_sort = self.get_feature()
        return c_df_sort.values[:,0]

#    def plot_variance_(self):
#        explained_variance_ = self.explained_variance_()
#        plt.figure()
#        plt.plot(explained_variance_, 'k')
#        plt.xlabel('n_components', fontsize = 14)
#        plt.ylabel('explained_variance_', fontsize = 14)
#        plt.show()

    def reduce_dimension(self):
        """Customer the n_components and calculate the contribution"""
        c_df_sort = self.get_feature()
        variance = self.explained_variance_()

        if self.n_components:
            p = c_df_sort.values[0:self.n_components,1:]
            y = np.dot(p, np.transpose(self.x))
            return np.transpose(y)

        variance_sum = sum(variance)
        variance_ratio = variance/variance_sum

        variance_contribution = 0
        for R in np.arange(self.dimension):
            variance_contribution += variance_ratio[R]
            if variance_contribution >= 0.99:
                break

        p = c_df_sort.values[0:R+1, 1:]
        y = np.dot(p, np.transpose(self.x))
        return np.transpose(y)

digits = datasets.load_digits()
x = digits.data
y = digits.target

if __name__ == '__main__':
    pca = PCA(x)
    y = pca.reduce_dimension()
