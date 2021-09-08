## Loading required packages
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

base_path = '/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Summer/ISYE6740 Machine Learning/Project/Trafficsigns'

## Reading the meta dataset
meta_data = pd.read_csv("Data/Meta.csv")
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
    meta_class.append(file.split('.')[0]) #title extracted
    
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
#%%
    #### Visualizing images representing the 43 classes in the data set

fig = plt.figure(figsize=(20, 5))
columns = 15
rows = 3

i=0
for c, img in zip(meta_class, meta_img):
    sp = fig.add_subplot(rows, columns, i+1)
    sp.set_title(f" Class {c}", fontdict = {'fontsize':8})  
    plt.imshow(img)
    i += 1
plt.show() 
fig.tight_layout() 

#### Understanding distribution of class frequencies in both the test and train data sets

train_dist = dict(Counter(train_labels))
test_dist = dict(Counter(test_labels))

x_train,y_train = zip(*sorted(train_dist.items()))
x_test,y_test = zip(*sorted(test_dist.items()))

def class_dist(x_train, y_train, x_test, y_test):

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(x_train, y_train)
    plt.axhline(y=np.average(y_train), color='r', linestyle='dashed', label="Mean")
    plt.axhline(y=np.median(y_train), color='y', linestyle='dashed', label="median")
    plt.xlabel("Class ID")
    plt.xticks(range(0, 45,5))
    plt.ylabel("Class Frequency")
    plt.title(f"Class Distribution of {sum(train_dist.values())} Training Images")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(x_test, y_test)
    plt.axhline(y=np.average(y_test), color='r', linestyle='dashed', label="Mean")
    plt.axhline(y=np.median(y_test), color='y', linestyle='dashed', label="median")
    plt.xlabel("Class ID")
    plt.xticks(range(0, 45,5))
    plt.ylabel("Class Frequency")
    plt.title(f"Class Distribution of {sum(test_dist.values())} Test Images")
    plt.legend()
    plt.show()

class_dist(x_train, y_train, x_test, y_test)


train_pix = dict(Counter([img.shape[0]*img.shape[1]*img.shape[2] for img in train_data_raw]))
test_pix = dict(Counter([img.shape[0]*img.shape[1]*img.shape[2] for img in test_data_raw]))

pixel_range = [(1000, 3000), (3000, 6000), (6000, 9000), (9000, 12000), (12000, 15000), (15000, 18000), (18000)]

def pixel_dist(train_pix, test_pix, pixel_range):
    pixel_dist = []

    fig = plt.figure(figsize=(15, 4))

    for tup in pixel_range:
        if isinstance(tup,int):
            pixel_dist.append(sum([v for k,v in train_pix.items() if (k >= tup)]))
        else:
            pixel_dist.append(sum([v for k,v in train_pix.items() if (k >= tup[0] and k < tup[1])]))

    plt.subplot(1, 2, 1)
    plt.bar([f'<{i[1]}' if isinstance(i,tuple) else f' >={i}' for i in pixel_range], pixel_dist)
    plt.xlabel("Pixel Range")
    plt.ylabel("Frequency of Train Images")
    plt.title("Distribution of Training Image Resolution")

    pixel_dist = []
    for tup in pixel_range:
        if isinstance(tup,int):
            pixel_dist.append(sum([v for k,v in test_pix.items() if (k >= tup)]))
        else:
            pixel_dist.append(sum([v for k,v in test_pix.items() if (k >= tup[0] and k < tup[1])]))

    plt.subplot(1, 2, 2)
    plt.bar([f'<{i[1]}' if isinstance(i,tuple) else f' >={i}' for i in pixel_range], pixel_dist)
    plt.xlabel("Pixel Range")
    plt.ylabel("Frequency of Test Images")
    plt.title("Distribution of Test Image Resolution")

    plt.show()

pixel_dist(train_pix, test_pix, pixel_range)

#%%
#Grouping the classes into 6 categories from 43 categories
# [ ]
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


# Data Pre-processing
# Applying below pre-processing techniques
# Resizing the data to 30 * 30 * 3 dimensions to have a common resolution for modeling (Already done and loaded in train_data and test_data)
# Scaling the data by diving by 255.
# Resampling the data to level the class imbalance.
    
## Preprocessing data with 2700 pixels
## Scaling the train data
train_arr = np.array(train_data)
train_arr = train_arr.reshape((train_arr.shape[0], 30*30*3))
train_data_scaled = train_arr.astype(float)/255

## Scaling the test data
test_arr = np.array(test_data)
test_arr = test_arr.reshape((test_arr.shape[0], 30*30*3))
test_data_scaled = test_arr.astype(float)/255

# [ ]
## Preprocessing data with 900 pixels considering only gray scale
## Scaling the train data
train_arr = np.array(train_data)
train_arr = np.mean(train_arr, -1)
train_arr = train_arr.reshape((train_arr.shape[0], 30*30))
train_data_scaled = train_arr.astype(float)/255

## Scaling the test data
test_arr = np.array(test_data)
test_arr = np.mean(test_arr, -1)

#%% MODELLING

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(train_data_scaled,train_labels, test_size=0.2, random_state = 24)

#DOWNSAMPLING
# X_Sampled, y_Sampled = rus.fit_resample(X_train, y_train)
start = time.time()

#neural network - MLP
clf_nn= MLPClassifier(solver='sgd',activation='relu', alpha=1e-5, hidden_layer_sizes=(43, 10), random_state=1).fit(X_train,y_train)

from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

nn_pred = clf_nn.predict(X_test/255)
nn_report = classification_report(y_test,nn_pred)
nn_confusion = confusion_matrix(y_test,nn_pred)
print(accuracy_score(y_test, nn_pred))

end= time.time()

#%% 
start = time.time()
model2 = RandomForestClassifier(n_estimators=500,max_features='sqrt')
model2.fit(X_train, y_train)
end = time.time()
print("Time Elapsed : ", end-start)

model2_pred = model2.predict(X_test/255)
model2_report = classification_report(y_test,model2_pred)
model2_confusion = confusion_matrix(y_test,model2_pred)
print(accuracy_score(y_test, model2_pred))
