#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:38:04 2021

@author: shanmukhateja
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap


marriage = pd.read_csv('marriage.csv', names= ["Atr1","Atr2","Atr3","Atr4","Atr5","Atr6","Atr7","Atr8","Atr9","Atr10","Atr11","Atr12","Atr13","Atr14","Atr15","Atr16","Atr17","Atr18","Atr19","Atr20","Atr21","Atr22","Atr23","Atr24","Atr25","Atr26","Atr27","Atr28","Atr29","Atr30","Atr31","Atr32","Atr33","Atr34","Atr35","Atr36","Atr37","Atr38","Atr39","Atr40","Atr41","Atr42","Atr43","Atr44","Atr45","Atr46","Atr47","Atr48","Atr49","Atr50","Atr51","Atr52","Atr53","Atr54","Class"])

"""
Check data & Split data for training and test
"""
marriage.head()

X = marriage.drop('Class',axis=1)
y = marriage['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=111)

"""
Defining the results output
"""
def clfresult(clf_fit,name):
    # ## training error
    ypred_train = clf_fit.predict(X_train)
    matched_train = ypred_train == y_train
    acc_train = sum(matched_train)/len(matched_train)
    acc_train
    
    # training confusion matrix
    idx0 = np.where(y_train ==0) 
    idx1 = np.where(y_train ==1)
    cf_train_00 = np.sum(y_train.to_numpy()[idx0] == ypred_train[idx0])
    cf_train_01 = np.sum(ypred_train[idx0] ==1)
    cf_train_11 = np.sum(y_train.to_numpy()[idx1] == ypred_train[idx1])
    cf_train_10 = np.sum(ypred_train[idx1] ==0)
    
    # ## test error
    ypred_test = clf.predict(X_test)
    matched_test = ypred_test == y_test
    acc_test = sum(matched_test)/len(matched_test)
    acc_test
    #clf.score(X_test, y_test)
       
    # testing confusion matrix
    idx1 = np.where(y_test ==1) 
    idx0 = np.where(y_test ==0)
    cf_test_11 = np.sum(y_test.to_numpy()[idx1] == ypred_test[idx1])
    cf_test_10 = np.sum(ypred_test[idx1] ==0)
    cf_test_00 = np.sum(y_test.to_numpy()[idx0] == ypred_test[idx0])
    cf_test_01 = np.sum(ypred_test[idx0] ==1)
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(name)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('The training accuracy: '+ str(round(acc_train, 4)))
    print('confusion matrix for training:')
    print('          predected 0       predected 1')
    print(f"true 0        {cf_train_00}                {cf_train_01}")
    print(f"true 1        {cf_train_10}                {cf_train_11}")
    print('\nThe testing accuracy: '+ str(round(acc_test, 4)))
    print('confusion matrix for testing:')
    print('          predected 0       predected 1')
    print(f"true 0        {cf_test_00}                {cf_test_01}")
    print(f"true 1        {cf_test_10}                {cf_test_11}")
    
    return acc_test

""" 
Naive Bayes
"""
#Remark: Please note that, here, for Naive Bayes, this means that we have to estimate the variance for each individual feature from training data. When estimating the variance, if the variance is zero to close to zero (meaning that there is very little variability in the feature), you can set the variance to be a small number, e.g.,   = 10^-3. We do not want to have include zero or nearly variance in Naive Bayes.

#Calculating attributes' variances 
variances = pd.DataFrame(X).var()
# plt.plot(variances)
# plt.title("Variances of attributes")
# plt.show()
print("None of the variances of the features are close to 0. They are all more than "+ str(round(min(variances),3)))

clf = GaussianNB(var_smoothing=10**-3).fit(X_train, y_train)
clfresult(clf,"Naive Bayes")

""" 
Logistic Regression
""" 
clf = LogisticRegression(max_iter=200, solver='liblinear').fit(X_train, y_train)
clfresult(clf,"Logistic Regression")


""" 
KNN
""" 
clf = KNeighborsClassifier(3).fit(X_train,y_train)
clfresult(clf,"KNN")

# acc_test=[]
# for neighbors in range(1,100):
#     acc_test.append( clfresult(KNeighborsClassifier(neighbors).fit(X_train,y_train),"KNN")
#     )
# plt.plot(acc_test)
#The accuracy is similar for different number of neighbors

#%%
""" 
PCA
"""
pca = PCA(n_components=2)
X_r = pca.fit(X_train).transform(X_train)
Xtest_r = pca.fit(X_test).transform(X_test)

"""
Defining a plotting for PCA 
"""    


def clfplot(clf_fit,name,ypred_train,ypred_test):
    h=0.02  # step size in the mesh
    x_min, x_max = X_r[:, 0].min() - .5, X_r[:, 0].max() + .5
    y_min, y_max = X_r[:, 1].min() - .5, X_r[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax1 = plt.subplot(1,1,1)
    ax1.set_title("Input data (light = test)")  
    ax1.scatter(X_r[:, 0], X_r[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    ax1.scatter(Xtest_r[:, 0], Xtest_r[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_xticks(())
    ax1.set_yticks(())
    plt.figure()
    
    #For the classifier
    ax = plt.subplot(1,1,1)
    score = clf_fit.score(Xtest_r, y_test)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf_fit, "decision_function"):
        Z = clf_fit.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf_fit.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(X_r[:, 0], X_r[:, 1], c=ypred_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(Xtest_r[:, 0], Xtest_r[:, 1], c=ypred_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    return score

#Naive Bayes
clf = GaussianNB(var_smoothing=10**-3).fit(X_r, y_train)
ypred_train = clf.predict(X_r)
ypred_test = clf.predict(Xtest_r)
clfplot(clf,"Naive Bayes",ypred_train,ypred_test)

#The data itself is very linearly seperable after dimentionality reduction with PCA.
#The Naive Bayes classifier has a Smooth, Narrow and a relatively non-linear decision boundary and gives mostly clear  seperability. except for a few misclassificed points.

#Logistic Regression

clf = LogisticRegression(max_iter=200, solver='liblinear').fit(X_r, y_train)
ypred_train = clf.predict(X_r)
ypred_test = clf.predict(Xtest_r)
clfplot(clf,"Logistic Regression",ypred_train,ypred_test)

#Logistic coundary is linear boundary with seemingly larger margins. 

#KNN
clf = KNeighborsClassifier(3).fit(X_r,y_train)
ypred_train = clf.predict(X_r)
ypred_test = clf.predict(Xtest_r)
clfplot(clf,"KNN",ypred_train,ypred_test)

#The boundary is noisy but has a higher prediction accuracy.