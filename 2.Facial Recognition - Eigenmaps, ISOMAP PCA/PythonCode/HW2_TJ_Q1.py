
#%% Q1.1  
import numpy as np
import math
import matplotlib.pyplot as plt
# import scipy.io as spio
import scipy.sparse.linalg as ll
# import sklearn.preprocessing as skpp
from skimage.transform import resize
from PIL import Image

#Importing Images Subject 1
im_t1 = Image.open("subject01-test.gif") 
im1 = Image.open("subject01.glasses.gif") 
im2 = Image.open("subject01.happy.gif") 
im3 = Image.open("subject01.leftlight.gif") 
im4 = Image.open("subject01.noglasses.gif") 
im5 = Image.open("subject01.normal.gif") 
im6 = Image.open("subject01.rightlight.gif") 
im7 = Image.open("subject01.surprised.gif") 
im8 = Image.open("subject01.sad.gif") 
im9 = Image.open("subject01.sleepy.gif") 
im10 = Image.open("subject01.wink.gif") 

H= im1.size[0]//4
W=im1.size[1]//4
data1 = np.ndarray.flatten(np.asarray((im1.resize((H,W)))))

for i in [im2,im3,im4,im5,im6,im7,im8,im9,im10]: #Appending other images of the subject 
    ir = np.ndarray.flatten(np.asarray((i.resize((H,W)))))
    data1 = np.append(data1,ir.T)
    # print(ir)
data1= data1.reshape((10,4800)) #Data matrix of subject 1 images


#Importing Images Subject 2
im_t2 = Image.open("subject02-test.gif") 
im1_2 = Image.open("subject02.glasses.gif") 
im2_2 = Image.open("subject02.happy.gif") 
im3_2 = Image.open("subject02.leftlight.gif") 
im4_2 = Image.open("subject02.noglasses.gif") 
im5_2 = Image.open("subject02.normal.gif") 
im6_2 = Image.open("subject02.rightlight.gif")  
im7_2 = Image.open("subject02.sad.gif") 
im8_2 = Image.open("subject02.sleepy.gif") 
im9_2 = Image.open("subject02.wink.gif") 

data2 = np.ndarray.flatten(np.asarray((im1_2.resize((H,W)))))
for i in [im2_2,im3_2,im4_2,im5_2,im6_2,im7_2,im8_2,im9_2]: #Appending other images of the subject 
    ir = np.ndarray.flatten(np.asarray((i.resize((H,W)))))
    data2 = np.append(data2,ir.T)
    # print(ir)    
data2 = data2.reshape((9,4800)) #Data matrix of subject 2 images

del i, ir,im1,im2,im3,im4,im5,im6,im7,im8,im9,im10
del im1_2,im2_2,im3_2,im4_2,im5_2,im6_2,im7_2,im8_2,im9_2

###     PCA Subject 1 
m,n = data1.shape
mu = np.mean(data1,axis = 0) #Mean 
xc = data1 - mu[None,:] #x-mu
C = np.dot(xc,xc.T)/n   # Covariance Matrix
K = 6 #Number of principal components
S,Wt = ll.eigs(C,k = K) #EIgen decomposition
S = S.real  # Top k Largest variances
Wt = Wt.real #Weights/Projection Directions

dim1 = np.dot(Wt[:,0].T,xc)/math.sqrt(S[0]) # extract 1st eigenvalues
dim2 = np.dot(Wt[:,1].T,xc)/math.sqrt(S[1]) # extract 2nd eigenvalue
dim3 = np.dot(Wt[:,2].T,xc)/math.sqrt(S[2]) # extract 1st eigenvalues
dim4 = np.dot(Wt[:,3].T,xc)/math.sqrt(S[3]) # extract 2nd eigenvalue
dim5 = np.dot(Wt[:,4].T,xc)/math.sqrt(S[4]) # extract 1st eigenvalues
dim6 = np.dot(Wt[:,5].T,xc)/math.sqrt(S[5]) # extract 2nd eigenvalue

#Eigen Faces Subject 1
plt.imsave('s1_ef1.png',dim1.reshape(W,H),cmap='gray')
plt.imsave('s1_ef2.png',dim2.reshape(W,H),cmap='gray')
plt.imsave('s1_ef3.png',dim3.reshape(W,H),cmap='gray')
plt.imsave('s1_ef4.png',dim4.reshape(W,H),cmap='gray')
plt.imsave('s1_ef5.png',dim5.reshape(W,H),cmap='gray')
plt.imsave('s1_ef6.png',dim6.reshape(W,H),cmap='gray')


###     PCA Subject 2
m2,n2 = data2.shape
mu2 = np.mean(data2,axis = 0) #Mean 
xc2 = data2 - mu2[None,:] #x-mu
C2 = np.dot(xc2,xc2.T)/n2   # Covariance Matrix
K = 6 #Number of principal components
S2,Wt2 = ll.eigs(C2,k = K) #EIgen decomposition
S2 = S2.real  # Top k Largest variances
Wt2 = Wt2.real #Weights/Projection Directions

dim12 = np.dot(Wt2[:,0].T,xc2)/math.sqrt(S2[0]) # extract 1st eigenvalues
dim22 = np.dot(Wt2[:,1].T,xc2)/math.sqrt(S2[1]) # extract 2nd eigenvalue
dim32 = np.dot(Wt2[:,2].T,xc2)/math.sqrt(S2[2]) # extract 1st eigenvalues
dim42 = np.dot(Wt2[:,3].T,xc2)/math.sqrt(S2[3]) # extract 2nd eigenvalue
dim52 = np.dot(Wt2[:,4].T,xc2)/math.sqrt(S2[4]) # extract 1st eigenvalues
dim62 = np.dot(Wt2[:,5].T,xc2)/math.sqrt(S2[5]) # extract 2nd eigenvalue

#Eigen Faces Subject 2
plt.imsave('s2_ef1.png',dim12.reshape(W,H),cmap='gray')
plt.imsave('s2_ef2.png',dim22.reshape(W,H),cmap='gray')
plt.imsave('s2_ef3.png',dim32.reshape(W,H),cmap='gray')
plt.imsave('s2_ef4.png',dim42.reshape(W,H),cmap='gray')
plt.imsave('s2_ef5.png',dim52.reshape(W,H),cmap='gray')
plt.imsave('s2_ef6.png',dim62.reshape(W,H),cmap='gray')

## Please explain can you see any patterns in the top 6 eigenfaces?
# The eignen faces capture different aspects of the pictures like below:
# For s1; ef1 captured most deviation from the mean in terms of overall face features, ef2 captured lighting from left, other eig faces captured combinations of multiple face expressions and lighting. 
# As expected The quality of features are higher/better in the order of the eigen values. 

 #%% Q1.2

#Vectorising downsized test images of subject 1 and subject 2 resp
t1 = np.ndarray.flatten(np.asarray((im_t1.resize((H,W)))))
t2 = np.ndarray.flatten(np.asarray((im_t2.resize((H,W)))))
#Top eigen faces of subject 1 and subject 2 resp
e1=dim1 
e2 = dim12

#Calculating the residual between the test image (corrected for mean) and the image mapped by eigenface of different subjects.
s11 = np.linalg.norm(((t1-mu) - e1.dot(e1.T.dot(t1-mu))),ord=2) #Projection of Test Image 1 on eigen face 1
s21 = np.linalg.norm(((t1-mu2) - e2.dot(e2.T.dot(t1-mu2))),ord=2) #Projection of Test Image 1 on eigen face 2
s12 = np.linalg.norm(((t2-mu) - e1.dot(e1.T.dot(t2-mu))),ord=2) #Projection of Test Image 2 on eigen face 1
s22 = np.linalg.norm(((t2-mu2) - e2.dot(e2.T.dot(t2-mu2))),ord=2) #Projection of Test Image 2 on eigen face 2

#Lower the projection residual value, impplies that the eigen vector explains the features of the test vector better. Hence the test image is closer to the eigen face. On this note, based on the below matrix of Projection Residuals... 
print(round(s11), round(s12),round(s21),round(s22))

#t1 belongs to subject 1 
#t2 belongs to subject 2

# The face recognition works alright for the clear front views, but if we build the eigen faces from more high definition images and include more variations in the training data (i.e, with side sligtly side poses, flipped images, distorted, different scales etc), we can achieve better Face ID results. 

#%%











