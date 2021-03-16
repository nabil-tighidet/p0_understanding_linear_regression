import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


def cost(X, y, theta):
    m = X.shape[0]
    y_predict = np.dot(X,theta)
    cost = (1/(2*m))*sum(np.square((y_predict - y)))
    return cost

def gradient_descent(X,y,theta,alpha,n_iter):
    m = X.shape[0]
    cost_values = np.zeros(n_iter)
    for itr in range(n_iter):
        y_predict = np.dot(X,theta)
        cost_derivative_theta0 = (1/m)*sum((y_predict-y))
        cost_derivative_theta1 = (1/m)*sum((y_predict-y)*X[:,1])
        theta = [theta[0]-alpha*cost_derivative_theta0, theta[1]-alpha*cost_derivative_theta1]
        cost_values[itr] = cost(X,y,theta)
    return [theta,cost_values]


def gradient_descent_multiple(X,y,theta,alpha,n_iter):
    m = X.shape[0]
    cost_values = np.zeros(n_iter)
    for itr in range(n_iter):
        y_predict = np.dot(X,theta)
        cost_derivative_theta0 = (1/m)*sum((y_predict-y))
        cost_derivative_theta1 = (1/m)*sum((y_predict-y)*X[:,1])
        cost_derivative_theta2 = (1/m)*sum((y_predict-y)*X[:,2])
        theta = [theta[0]-alpha*cost_derivative_theta0, theta[1]-alpha*cost_derivative_theta1,theta[2]-alpha*cost_derivative_theta2]
        cost_values[itr] = cost(X,y,theta)
    return [theta,cost_values]


        
        