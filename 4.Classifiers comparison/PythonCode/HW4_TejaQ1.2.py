#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:28:47 2021

@author: shanmukhateja
"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

digits = loadmat("mnist_10digits.mat")
xtrain = digits['xtrain']
xtest = digits['xtest']
ytrain = digits['ytrain'].ravel()
ytest = digits['ytest'].ravel()
#%%
plt.hist(ytrain)
plt.hist(ytest)
plt.title("Sampling distribution")
#We have even samples of all digits in both train and test data

math.sqrt(xtrain.shape[1])

#logistic regression
clf_logistic = LogisticRegression(max_iter=200, solver='liblinear').fit(xtrain/255, ytrain)
logistic_pred = clf_logistic.predict(xtest/255)
log_report = classification_report(ytest,logistic_pred)
log_confusion = confusion_matrix(ytest,logistic_pred)
accuracy_score(ytest, logistic_pred)

#neural networks
clf_nn= MLPClassifier(solver='lbfgs',activation='relu', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1).fit(xtrain/255,ytrain)
nn_pred = clf_nn.predict(xtest/255)
nn_report = classification_report(ytest,nn_pred)
nn_confusion = confusion_matrix(ytest,nn_pred)
accuracy_score(ytest, nn_pred)

#Downsampling
ind= np.argwhere(ytrain==0)[0:500]
for i in range(1,10):
    ind = np.concatenate((ind, np.argwhere(ytrain==i)[0:500]))
ind=ind.ravel()
xtrain_d = xtrain[ind]
ytrain_d = ytrain[ind]


#KNN
k=71
clf_knn = KNeighborsClassifier(k).fit(xtrain_d/255, ytrain_d)
knn_pred = clf_knn.predict(xtest/255)
knn_report = classification_report(ytest,knn_pred)
knn_confusion = confusion_matrix(ytest,knn_pred)
accuracy_score(ytest, knn_pred)

#SVM 
clf_svm = SVC(kernel= 'linear').fit(xtrain_d/255,ytrain_d)
svm_pred = clf_svm.predict(xtest/255)
svm_report = classification_report(ytest,svm_pred)
svm_confusion = confusion_matrix(ytest,svm_pred)
accuracy_score(ytest, svm_pred)

#kernel SVM (radial)
clf_svm2 = SVC(kernel= 'rbf').fit(xtrain_d/255,ytrain_d)
svm_pred2 = clf_svm2.predict(xtest/255)
svm_report2 = classification_report(ytest,svm_pred2)
svm_confusion2 = confusion_matrix(ytest,svm_pred2)
accuracy_score(ytest, svm_pred2)


