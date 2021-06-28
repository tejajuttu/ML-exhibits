#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:00:46 2021

@author: shanmukhateja
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix 
from scipy import spatial
from PIL import Image
from numpy import asarray
import random
import time
import pandas as pd
#------------------------#

img = asarray(Image.open('GeorgiaTech.bmp'))  # load the image & convert image to numpy array
clusters = 16

def compress_img(img, clusters):
    random.seed(100) 
    start_time = time.time()
    H= img.shape[0] #height of image
    W=img.shape[1] #Width of image
    pixels= H*W ; print("Pixels = %d" %pixels)   # H x W pixels and 3 colors (RBG) for each pixel
    #plt.imshow(img) # plotting the given image 
    
    #Flatenning the image arrays to 2D 
    img2d = img.reshape((-1,3))
    m,n = img2d.shape #m= data points n= dimentions (3 for colored, 1 for B/W)
    #img2d = np.reshape(img, (H * W, img.shape[2]))
    
    

    # Random initialization of centroids from the existing pixels 
    centroids = img2d[random.sample(range(m),clusters),:]
    
    #Poor initialization of centroids from RGB colorspace
    # centroids = np.random.randint(0,255, size=(clusters,n))
   

    #print("Random centroids initialization")
    #print(centroids) #Cluster centers chosen    
    iter_max = 500     #SET
    count = np.empty([1,clusters])
    for iter in range(0, iter_max):
        count_old = count.copy()
        centroids_old = centroids.copy()

        #print("--iteration %d \n" % iter)
 
        #--------------------------------------------------------
    
        # # norm squared of the centroids;
        # For each data point x, computer min_j  -2 * x' * c_j + c_j^2;
        # Note that here is implemented as max, so the difference is negated.
        # c2 = np.sum(np.power(centroids.T, 2), axis=0, keepdims=True)  
        # tmpdiff = (2 * np.dot(img2d, centroids.T) - c2)
        # labels = np.argmax(tmpdiff, axis=1)
        # print(labels)
            
        # #Calculating the distance matrix between each pixel of the image and centroids chosen
        #cityblock implies manhattan distance
        tmpdiff = spatial.distance.cdist(img2d, centroids, 'cityblock')      
        labels = np.argmin(tmpdiff, axis=1)

        # #Alter commenting between the above two
        # #--------------------------------------------------------
    
        # #Recomputing the cluster centers by Updating the data assignment matrix;
        # # The assignment matrix is a sparse matrix, with size m x cluternumbers.
        # P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, clusters))
        # count = P.sum(axis=0)
        # #print(count)
        
        # # P*img2d implements summation of data points assigned to a given cluster.
        # centroids = np.array((P.T.dot(img2d)) / count.T)
        # #Initialising empty centroids with black pixels RGB000
        # centroids = np.nan_to_num(centroids) 
        # #print(centroids)
           
        for j in range(0,clusters):
            centroids[j,:] = np.median(img2d[labels == j,:], axis=0)        
            
        if np.array_equal(centroids_old,centroids):
            total_iters = iter
            break
        total_iters = iter
        
    #Taking the respectively labelled cluster centers for each of the m pixels    
    compressed = centroids[labels].astype(np.uint8) #converting to int
    imgout = np.reshape(compressed, (H, W,img.shape[2])) #reshaping back to 3D
    plt.imshow(imgout) # plotting the compressed image 
    
    runtime=(time.time() - start_time)
    print("Clusters --> %d" %clusters)
    print("--- Runtime %s seconds --- " % runtime)
    print ("--- Iterations to converge ---> %d"%total_iters)
    print("---- Compressed Image %d clusters---> "%clusters)
    
    plt.show()
    # plt.imsave('1.3earthcompressed_' + str(clusters) +'clusters_'+str(total_iters)+'iters.png', imgout)
    return (runtime,total_iters)
    
     #SET INPUTS TO CALLING THE FUNCTION DEFINED ABOVE
img1 = asarray(Image.open('GeorgiaTech.bmp'))  # load the image & convert image to numpy array
summary_img1 = pd.DataFrame([[0, 0, 0]],columns=["clusters","Time","Iterations"])
for clusters in [2,4,8,16]:
    t1,i1 =compress_img(img1,clusters)   
    summary_img1 = summary_img1.append({'clusters' : clusters,'Time' : t1, 'Iterations':i1} , ignore_index=True)
print("Image1 = GeorgiaTech.bmp")
print(summary_img1)
    

img2 = asarray(Image.open('football.bmp'))  #SET
summary_img2 = pd.DataFrame([[0, 0, 0]],columns=["clusters","Time","Iterations"])
for clusters in [2,4,8,16]:
    t2,i2 =compress_img(img2,clusters)   
    summary_img2 = summary_img2.append({'clusters' : clusters,'Time' : t2, 'Iterations':i2} , ignore_index=True)
print("Image2 = football.bmp")
print(summary_img2)


img3 = asarray(Image.open('earth.jpeg'))  #SET
summary_img3 = pd.DataFrame([[0, 0, 0]],columns=["clusters","Time","Iterations"])
for clusters in [2,4,8,16]:
    t3,i3 =compress_img(img3,clusters)   
    summary_img3 = summary_img3.append({'clusters' : clusters,'Time' : t3, 'Iterations':i3} , ignore_index=True)
print("Image3 = earth.jpeg")
print(summary_img3)


#HW 1 Summary tables for all pictures
print("\n Image1 = GeorgiaTech.bmp")
print(round (summary_img1,4).to_string(index=False))
print("\n Image2 = football.bmp")
print(round (summary_img2,4).to_string(index=False))
print("\n Image3 = earth.jpeg")
print(round (summary_img3,4).to_string(index=False))








