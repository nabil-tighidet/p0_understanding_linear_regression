import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


def cost(X, y, theta):
    m = X.shape[0] #taille de l'échantillon
    y_predict = np.dot(X,theta) #Calcul des prédictions
    cost = (1/(2*m))*sum(np.square((y_predict - y))) # Calcul du coût suivant la formule présentée dans le notebook.
    return cost

def gradient_descent(X,y,theta,alpha,n_iter):
    m = X.shape[0] # Taille de l'échantillon
    cost_values = np.zeros(n_iter) # Initialisation du vecteur qui va contenir l'historique des coûts
    for itr in range(n_iter): # Le nombre d'itération est un paramètre de notre fonction 
        y_predict = np.dot(X,theta) # Calcul des prédictions
        cost_derivative_theta0 = (1/m)*sum((y_predict-y)) # Calcul de la dérivée du coût par rapport à theta_0
        cost_derivative_theta1 = (1/m)*sum((y_predict-y)*X[:,1]) # Calcul de la dérivée du coût par rapport à theta_1
        theta = [theta[0]-alpha*cost_derivative_theta0, theta[1]-alpha*cost_derivative_theta1] # Mise à jour de la valeur des paramètres theta de la régression
        cost_values[itr] = cost(X,y,theta) # Stockage de la valeur du coût pour l'iteraction en cours
    return [np.array(theta),np.array(cost_values)]


def gradient_descent_multiple(X,y,theta,alpha,n_iter):
    m = X.shape[0] # Taille de l'échantillon
    cost_values = np.zeros(n_iter) # Initialisation du vecteur qui va contenir l'historique des coûts
    for itr in range(n_iter): # Le nombre d'itération est un paramètre de notre fonction 
        y_predict = np.dot(X,theta) # Calcul des prédictions
        cost_derivative_theta0 = (1/m)*sum((y_predict-y)) # Calcul de la dérivée du coût par rapport à theta_0
        cost_derivative_theta1 = (1/m)*sum((y_predict-y)*X[:,1]) # Calcul de la dérivée du coût par rapport à theta_1
        cost_derivative_theta2 = (1/m)*sum((y_predict-y)*X[:,2]) # Calcul de la dérivée du coût par rapport à theta_2
        theta = [theta[0]-alpha*cost_derivative_theta0, theta[1]-alpha*cost_derivative_theta1,theta[2]-alpha*cost_derivative_theta2] # Mise à jour de la valeur des paramètres theta de la régression
        cost_values[itr] = cost(X,y,theta) # Stockage de la valeur du coût pour l'iteraction en cours
    return [np.array(theta),np.array(cost_values)]

def reverse_normalization(theta_nmz,X):
    # Cette fonction recalcule les paramètres theta_i en tenant compte de la normalisation appliquée aux variables pour optimiser le gradient descendant.
    n = X.shape[1]-1 #Nombre de paramètres theta_i
    theta_0 = theta_nmz[0] #Initialisation de theta_0
    for i in range(n): #Initialisation de theta_0
        theta_0 = theta_0-(theta_nmz[i+1]*np.mean(X[:,i+1]))/(np.max(X[:,i+1])-np.min(X[:,i+1])) # Calcul de theta_0 suivant la formule présenté dans le notebook.
    theta=X[:,1:n+1].max(0)-X[:,1:n+1].min(0) # Calcul des autres theta_i suivant la formule présenté dans le notebook.
    theta=[1/x for x in theta] 
    theta=theta_nmz[1:n+1]*theta 
    theta = np.insert(theta, 0, theta_0, axis=0) # Concaténation de theta_0 et des autres theta_i dans un même vecteur theta
    return theta                   

        
        