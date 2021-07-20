#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 19:14:20 2021

@author: shanmukhateja
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as spio
import sklearn.preprocessing as skpp
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.utils.graph_shortest_path import graph_shortest_path
import scipy.sparse.linalg as ll
import random

#Settings and Tunings
faces = spio.loadmat('data/isomap.mat',squeeze_me=True)
imgdata=faces['images']
eps = 15
# eps = 800
K =2
random.seed(100)

#ISO MAP Multi-Dimensional Scaling
n, m = imgdata.shape
dist = cdist(imgdata.T, imgdata.T, metric='euclidean') #Computing Pairwise shortest distance matrix
# dist = cdist(imgdata.T, imgdata.T, metric='cityblock') #Computing Pairwise shortest distance matrix
A =  np.zeros((m, m)) + np.inf          
aij = dist < eps                #Adding weights (epsilon) to capture local geometry of the datapoints
A[aij] = dist[aij]          #Weighted Adjacency Matrix
plt.imshow(A)
#plt.hist(A)
plt.imsave('Adj_Nearest Neighbor graph.png',A)

import PIL
PIL.Image.fromarray(A,'')

D = graph_shortest_path(A) #DISTANCE MATRIX

H = np.eye(m) - (1/m)*np.ones((m, m))
D2 = D**2
C = -1/(2*m) * H.dot(D2).dot(H) #Centring Matrix

S,Wt = ll.eigs(C,k = K) #EIgen decomposition
z = Wt.dot(np.diag(S**(1/2)))
print(z.real) #(ndarray) the new reduced matrix(m x 2)

plt.scatter(z[:, 0], z[:, 1], marker='.')

   
#PLOTTING THE GRAPH

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
ax.set_title('Facial Netowork Chart')
ax.set_xlabel('Dimension: 1')
ax.set_ylabel('Dimension: 2')
#setting size of the plots
x_size = (max(z[:, 0]) - min(z[:, 0])) * 0.08
y_size = (max(z[:, 1]) - min(z[:, 1])) * 0.08

# Show 50 images of the given image dataset on to the plot
for i in range(50):
    random.seed(100)
    img_num = np.random.randint(0, m)
    x0 = z[img_num, 0] - (x_size / 2.)
    y0 = z[img_num, 1] - (y_size / 2.)
    x1 = z[img_num, 0] + (x_size / 2.)
    y1 = z[img_num, 1] + (y_size / 2.)
    img = imgdata[:, img_num].reshape(64, 64).T
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    
    # Show 2D z plot
ax.scatter(z[:, 0], z[:, 1], marker='.',alpha=0.7)

ax.set_ylabel('Top-Down Bright-Dark pixels')
ax.set_xlabel('Right-Left dark-bright pixels and face Pose')

plt.savefig('outISOMAP.png')
# plt.savefig('outISOMAP_manhattan.png')

#%% PART 3

                    # PCA
m,n = imgdata.T.shape
mu = np.mean(imgdata.T,axis = 0) #Mean 
xc = imgdata.T - mu[None,:] #x-mu
C_pca = np.dot(xc,xc.T)/m   # Covariance Matrix
K = 2 #Number of principal components
S_pca,Wt_pca = ll.eigs(C_pca,k = K) #EIgen decomposition
S_pca = S_pca.real  # Top k Largest variances
Wt_pca = Wt_pca.real #Weights/Projection Directions

zpca = Wt_pca.dot(np.diag(S_pca**(-1/2)))
print(zpca.real) #(ndarray) the new reduced matrix(m x 2)

plt.scatter(zpca[:, 0], zpca[:, 1], marker='.')

   
#PLOTTING THE GRAPH

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
ax.set_title('PCA Based - Facial Netowork Chart')
ax.set_xlabel('Dimension: 1')
ax.set_ylabel('Dimension: 2')
#setting size of the plots
x_size = (max(zpca[:, 0]) - min(zpca[:, 0])) * 0.08
y_size = (max(zpca[:, 1]) - min(zpca[:, 1])) * 0.08

# Show 50 images of the given image dataset on to the plot
for i in range(40):
    random.seed(100)
    img_num = np.random.randint(0, m)
    x0 = zpca[img_num, 0] - (x_size / 2.)
    y0 = zpca[img_num, 1] - (y_size / 2.)
    x1 = zpca[img_num, 0] + (x_size / 2.)
    y1 = zpca[img_num, 1] + (y_size / 2.)
    img = imgdata[:, img_num].reshape(64, 64).T
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    
    # Show 2D zpca plot
ax.scatter(zpca[:, 0], zpca[:, 1], marker='.',alpha=0.7)

ax.set_ylabel('Top-Down NA')
ax.set_xlabel('Right-Left NA')

plt.savefig('outpca.png')

#%%
"""    plotting code inspired by:
        http://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/
"""