#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:16:32 2021

@author: shanmukhateja
"""
import pandas as pd
import numpy as np
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
import math

df = pd.read_csv('data/food-consumption.csv')
data= df.to_numpy()

labels = data[:,0]
food = list(df)[1:]
data = data[:,1:21].astype(float)


m,n = data.shape
mu = np.mean(data,axis = 1) #Each country's Mean food consumption
xc = data.T - mu #x-mu
C_pca = np.dot(xc,xc.T)/m   # Covariance Matrix
K = 2 #Number of principal components
S ,Wt_pca = ll.eigs(C_pca,k = K) #EIgen decomposition
S = S.real  # Top k Largest variances
Wt_pca = Wt_pca.real #Weights/Projection Directions

z= np.dot(Wt_pca.T,xc)/math.sqrt(S[0])

plt.scatter(z[0], z[1], marker='.')

for i, txt in enumerate(labels):
    plt.annotate(txt, (z[0,i], z[1,i]))
    
#%%
    
m2,n2 = data.T.shape
mu2 = np.mean(data.T,axis = 1) #Each country's Mean food consumption
xc2 = data - mu2 #x-mu
C_pca2 = np.dot(xc.T,xc)/m2   # Covariance Matrix
K = 2 #Number of principal components
S2 ,Wt2 = ll.eigs(C_pca2,k = K) #EIgen decomposition
S2 = S2.real  # Top k Largest variances
Wt2 = Wt2.real #Weights/Projection Directions

z2= np.dot(Wt2.T,xc2)/math.sqrt(S2[0])

plt.scatter(z2[0], z2[1], marker='.')

for i, txt in enumerate(food):
    plt.annotate(txt, (z2[0,i], z2[1,i]))