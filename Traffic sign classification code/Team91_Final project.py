#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 20:22:07 2021

@author: shanmukhateja
"""


import numpy as np
import pandas as pd
import time
import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

#Models
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC , NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc



base_path = '/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Summer/ISYE6740 Machine Learning/Project/Trafficsigns'

## Reading the meta dataset
meta_data = pd.read_csv(f"{base_path}/Data/Meta.csv")
## Getting the number of classes from the Meta dataset
classes = meta_data.shape[0]

## Loading the sample image for each class from Meta dataset
meta_img = []
meta_class = []
meta_path = f'{base_path}/Data/Meta/'
meta_files = os.listdir(meta_path)
meta_files=[x for x in meta_files if x.endswith('.png')]

for file in meta_files:
    '''
    Note: I am not resizing these images as I will use them for EDA purposes only.
    '''
    image = cv2.imread(meta_path+file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    meta_img.append(image)
    meta_class.append(file.split('.')[0])
    
## Reading the Train data set (consists of images in multiple folders).This piece of code will iteratively read images from every folder.
'''
Two lists are populated. The raw training images are of different resolutions. They are loaded as such in train_data_raw. 
For purpose of modeling, all images are resized to a common resolution (30x30) and loaded into train_data
'''

train_data_raw = []
train_data=[]
train_labels=[]

res = 30

for c in range(classes) :
    path = f'{base_path}/Data/Train/{c}/'.format(c)
    files = os.listdir(path)
    files=[x for x in files if x.endswith('.png')]

    for file in files:
        train_image = cv2.imread(path+file)
        train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(train_image, (res, res), interpolation = cv2.INTER_AREA)
        train_data.append(np.array(image_resized))
        train_data_raw.append(train_image)
        train_labels.append(c)
        
## Reading the Test data images 
test_csv = pd.read_csv(f'{base_path}/Data/Test.csv')
test_img_path = test_csv['Path']

## List containing class labels for test data
test_labels = test_csv['ClassId'].values
test_data = []  ## List to hold resized test images
test_data_raw = []  ## List to hold test images in raw format

for f in test_img_path:
    test_image = cv2.imread(f'{base_path}/Data/' + f)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(test_image, (res, res), interpolation = cv2.INTER_AREA)
    test_data.append(np.array(image_resized))
    test_data_raw.append(test_image)
    
    #from PIL import Image
    #image_from_array = Image.fromarray(image, 'RGB')
    #resized_image = np.array(image_from_array.resize((30, 30)))

#### Grouping the classes into 6 categories from 43 categories

## Creating groups of classes and getting indexes of group elements from test and train datatsets.
groups = {'speed':[0,1,2,3,4,5,7,8], 'prohibitory':[9,10,15,16], 'derestriction':[6,32,41,42], 'mandatory':[33,34,35,36,37,38,39,40], 'danger':[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31], 'other':[12,14,13,17]}
group_labels = {'speed':1, 'prohibitory':2, 'derestriction':3, 'mandatory':4, 'danger':5, 'other':6}

group_indices_train = {}
group_indices_test = {}

for k in groups.keys():
    group_indices_train[k] = np.where(np.isin(np.array(train_labels), groups[k]) == True)[0]
    group_indices_test[k] = np.where(np.isin(np.array(test_labels), groups[k]) == True)[0]
    
## Creating new labels for train and test datasets
train_lbl_grp = np.array(train_labels)
test_lbl_grp = np.array(test_labels)

## Updating new label values for each group
for k,v in group_labels.items():  ##New group labels
    train_lbl_grp[group_indices_train[k]] = v
    test_lbl_grp[group_indices_test[k]] = v
    
    
#### Applying below pre-processing techniques

# 1. Resizing the data to 30 * 30 * 3 dimensions to have a common resolution for modeling (Already done and loaded in train_data and test_data)
# 2. Scaling the data by diving by 255. 
# 3. Resampling the data to level the class imbalance.

## Preprocessing data with 2700 pixels
## Scaling the train data
train_arr = np.array(train_data)
train_arr = train_arr.reshape((train_arr.shape[0], 30*30*3))
train_data_scaled = train_arr.astype(float)/255

## Scaling the test data
test_arr = np.array(test_data)
test_arr = test_arr.reshape((test_arr.shape[0], 30*30*3))
test_data_scaled = test_arr.astype(float)/255

## Labels : train_lbl_grp, test_lbl_grp

## Preprocessing data with 900 pixels considering only gray scale
## Scaling the train data
train_arr = np.array(train_data)
train_arr = np.mean(train_arr, -1)
train_arr = train_arr.reshape((train_arr.shape[0], 30*30))
train_data_scaled = train_arr.astype(float)/255

## Scaling the test data
test_arr = np.array(test_data)
test_arr = np.mean(test_arr, -1)
test_arr = test_arr.reshape((test_arr.shape[0], 30*30))
test_data_scaled = test_arr.astype(float)/255

## Labels : train_lbl_grp, test_lbl_grp
#%% MODELLING - RandomForest 

import time
## Splitting into test/validation
X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_lbl_grp, test_size=0.2, random_state = 24)


start = time.time()

rus = RandomUnderSampler(random_state=0)
X_Sampled, y_Sampled = rus.fit_resample(X_train, y_train)

# define models and parameters
model = RandomForestClassifier()
n_estimators = [100, 300, 500, 700, 900,1000]
max_features = ['sqrt','log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_Sampled, y_Sampled)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

end = time.time()

print(start-end)


start = time.time()
rf_model = RandomForestClassifier(n_estimators=900,max_features='sqrt')
rf_model.fit(X_train, y_train)
end = time.time()
print("Time Elapsed : ", end-start)

val_pred = rf_model.predict(X_test)
print("Vaidation accuracy for RF model is : ", accuracy_score(y_test,val_pred))

#TEST ACCURACY
test_pred = rf_model.predict(test_data_scaled)
print("Test accuracy for RF model is : ", accuracy_score(test_lbl_grp,test_pred))
print(confusion_matrix(test_lbl_grp, test_pred))
print(classification_report(test_lbl_grp, test_pred))


#%% NEURAL NETS

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import time
## Splitting into test/validation
# X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_labels, test_size=0.2, random_state = 24)
X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_lbl_grp, test_size=0.2, random_state = 24)


start = time.time()

#neural network - MLP
clf_nn= MLPClassifier(solver='sgd',activation='relu', alpha=1e-5, hidden_layer_sizes=(43, 15), random_state=1).fit(X_train,y_train)

# from keras.utils import to_categorical
# y_train = to_categorical(y_train, 43)
# y_test = to_categorical(y_test, 43)

nn_pred = clf_nn.predict(X_test)
nn_report = classification_report(y_test,nn_pred)
nn_confusion = confusion_matrix(y_test,nn_pred)
print(accuracy_score(y_test, nn_pred))
end= time.time()
print("Time taken for NLP is ", end-start)

#TEST ACCURACY
test_pred_nn = clf_nn.predict(test_data_scaled)
print("Test accuracy for Neural Network Classification model is : ", accuracy_score(test_lbl_grp,test_pred_nn))
print(confusion_matrix(test_lbl_grp, test_pred_nn))
print(classification_report(test_lbl_grp, test_pred_nn))


#%% SVM
from sklearn.svm import SVC , NuSVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import time
## Splitting into test/validation
# X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_labels, test_size=0.2, random_state = 24)
X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_lbl_grp, test_size=0.2, random_state = 24)


start = time.time()
#neural network - MLP

clf_svc= NuSVC(nu=0.05,gamma=0.00001, kernel= 'rbf').fit(X_train,y_train)

# from keras.utils import to_categorical
# y_train = to_categorical(y_train, 43)
# y_test = to_categorical(y_test, 43)

svc_pred = clf_svc.predict(X_test)
svc_report = classification_report(y_test,svc_pred)
svc_confusion = confusion_matrix(y_test,svc_pred)
print(accuracy_score(y_test, svc_pred))

end= time.time()
print("Time taken for SVM is ", end-start)

#TEST ACCURACY
test_pred_svc = clf_svc.predict(test_data_scaled)
print("Test accuracy for SVM model is : ", accuracy_score(test_lbl_grp,test_pred_svc))
print(confusion_matrix(test_lbl_grp, test_pred_svc))
print(classification_report(test_lbl_grp, test_pred_svc))

#%% KNN

from sklearn.neighbors import KNeighborsClassifier
import time
## Splitting into test/validation
# X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_labels, test_size=0.2, random_state = 24)
X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_lbl_grp, test_size=0.2, random_state = 24)



error_rate = []
# Will take some time
for i in range(2,10):
 
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(2,10),error_rate, color='blue', linestyle = 'dashed', marker = 'o', markersize=10, markerfacecolor='red')
plt.title("Error Rate vs. K Value")
plt.xlabel('K')
plt.ylabel('Error Rate')



start = time.time()
clf_knn= KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)

# from keras.utils import to_categorical
# y_train = to_categorical(y_train, 43)
# y_test = to_categorical(y_test, 43)

knn_pred = clf_knn.predict(X_test)
knn_report = classification_report(y_test,knn_pred)
knn_confusion = confusion_matrix(y_test,knn_pred)
print(accuracy_score(y_test, knn_pred))

end= time.time()
print("Time taken for KNN is ", end-start)

#TEST ACCURACY
test_pred_knn = clf_knn.predict(test_data_scaled)
print("Test accuracy for KNN model is : ", accuracy_score(test_lbl_grp,test_pred_knn))
print(confusion_matrix(test_lbl_grp, test_pred_knn))
print(classification_report(test_lbl_grp, test_pred_knn))

#%%% LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import time
## Splitting into test/validation
# X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_labels, test_size=0.2, random_state = 24)
X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_lbl_grp, test_size=0.2, random_state = 24)


start = time.time()
#neural network - MLP
clf_LDA = LinearDiscriminantAnalysis(n_components=5,solver = 'svd').fit(X_train,y_train)

# from keras.utils import to_categorical
# y_train = to_categorical(y_train, 43)
# y_test = to_categorical(y_test, 43)

LDA_pred = clf_LDA.predict(X_test)
LDA_report = classification_report(y_test,LDA_pred)
LDA_confusion = confusion_matrix(y_test,LDA_pred)
print(accuracy_score(y_test, LDA_pred))

end= time.time()
print("Time taken for LDA is ", end-start)

#TEST ACCURACY
test_pred_LDA = clf_LDA.predict(test_data_scaled)
print("Test accuracy for LDA model is : ", accuracy_score(test_lbl_grp,test_pred_LDA))
print(confusion_matrix(test_lbl_grp, test_pred_LDA))
print(classification_report(test_lbl_grp, test_pred_LDA))


