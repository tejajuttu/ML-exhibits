
import csv
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import seaborn as sns
import scipy


# load wine dataset
path = 'data/n90pol.csv'

n90pol = pd.read_csv(path,header=None).to_numpy()
n90pol.shape #90
data = n90pol[1:,0:2]
orientation = n90pol[1:,2].astype(int)
m,n = data.shape
plt.hist(n90pol[:,2])
plt.title('Observational Distribution of Political Inclination') 
plt.show()
# Observation: The data is more inclined towards liberal political orientation and no samples of being very conservative.
data = data.astype(float) # COnverting object to float data type

#1-D Histogram 
plt.hist(data[:,0],bins=10,alpha = 0.6, label  ="amygdala",rwidth=0.9)
plt.title('1-D Histogram of Amygdala')
plt.show()
plt.hist(data[:,1],bins=10,alpha  = 0.6, label = "acc", rwidth =0.9)
plt.title('1-D Histogram of acc')
plt.show()

binsize = m**(-1/3) #delta
n_bins = int(np.reciprocal(binsize)) #1/delta
plt.hist(data[:,0:2],bins=n_bins,alpha = 0.6, label  = ["amygdala","acc"])
plt.title('1-D Histogram of Amygdala & acc with 4 bins')
plt.legend()
plt.show()


# 1-D KDE
# for h in np.arange(0.008, 0.014, 0.002):
#     sns.kdeplot(data=data[:,0],bw=h,label="Bandwidth {:.3f}".format(h))
#     plt.title("Gaussian KDE of Amygdala")
# plt.figure()
# for h in np.arange(0.004, 0.01, 0.002):
#     sns.kdeplot(data=data[:,1],bw=h,label="Bandwidth {:.3f}".format(h))
#     plt.title("Gaussian KDE of acc")    

plt.figure()
bw1 = 1.06*np.std(data[:,0])*m**(-1/5) #Using Silverman's rule of thumb
print("For Amygdala, we will use the band width = {:.3f} ".format(bw1))
sns.kdeplot(data=data[:,0],bw=bw1,label="Amygdala BW={:.4f}".format(bw1))
plt.title("Gaussian KDE of Amygdala")

plt.figure()
bw2 = 0.008
bw2=1.06*np.std(data[:,1])*m**(-1/5) #Using Silverman's rule of thumb
print("For acc, we will use the band width = {:.3f} ".format(bw2))
sns.kdeplot(data=data[:,1],bw=bw2,label="acc BW={:.4f}".format(bw2))
plt.title("Gaussian KDE of acc")

   
#b. 2D countour plot of the 2D data
plt.figure()
plt.hist2d(data[:,0], data[:,1], bins=(15, 15), cmap=plt.cm.jet, label  = ["amygdala","acc"])
plt.colorbar()
plt.title("heatmap of 2-D histogram with 15 x 15 bins")  
plt.xlabel('amygdala')
plt.ylabel('acc')
plt.show()


# 3D plot for the 2 dimensional data
#Referring: DEmo code from class to build the 3-D plot of histogram
min_data = data.min(0)
max_data = data.max(0)
for nbin in [36]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=nbin)
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz )
    ax.set_xlabel('amygdala')
    ax.set_ylabel('acc')
    ax.set_zlabel('Count')
    ax.set_title('3D Histogram with {:.0f} bins'.format(nbin))
    plt.show()
print('Distribution is clear with 25 bins')

#c. 2D Gaussian KDE (e.g., heat-map, contour plot, etc.)
sns.kdeplot(data[:,0],data[:,1],cmap="Reds", shade=True, bw=bw1) #Silverman's bandwidth
sns.rugplot(data[:,0])
sns.rugplot(data[:,1],vertical=True)
plt.title('2D Gaussian KDE contour')
plt.xlabel('amygdala')
plt.ylabel('acc')
plt.show()


kde_amygdala = scipy.stats.gaussian_kde(data[:,0], bw_method='silverman')
pdf_amygdala = scipy.stats.gaussian_kde.pdf(kde_amygdala,data[:,0])
kde_acc = scipy.stats.gaussian_kde(data[:,1],bw_method='silverman')
pdf_acc = scipy.stats.gaussian_kde.pdf(kde_amygdala,data[:,1])

kde_joint = scipy.stats.gaussian_kde(data[:,0:2].T, bw_method='silverman')
pdf_joint = scipy.stats.gaussian_kde.pdf(kde_joint,data[:,0:2].T)

diff= np.multiply(pdf_amygdala,pdf_acc) - pdf_joint
plt.plot(abs(diff)/pdf_joint)
np.mean(abs(diff))/np.mean(pdf_joint)
#Difference is around 30% which says they may/may not be 

#d. Conditional KDE distributions
bw1 = bw1
for c in [2,3,4,5]:
    plt.figure()
    ppp=sns.kdeplot(data=data[orientation==c,0],bw=bw1,label="BW={:.3f}".format(bw1)) # THE PLot
    x,y = ppp.get_lines()[0].get_data()
    cdf = scipy.integrate.cumtrapz(y, x, initial=0)
    nearest_05 = np.abs(cdf-0.5).argmin()
    x_mean = x[nearest_05]
    y_mean = y[nearest_05]
    plt.vlines(x_mean, 0, y_mean)
    plt.title("Conditional Gaussian KDE of Amygdala with orientation = {}".format(c))
    plt.show()
    print('Median Amygdala value with orientation c={} is {:.5f}'.format(c,x_mean))
    print(np.mean(data[orientation==c,0]))

bw2 = bw2
for c in [2,3,4,5]:
    plt.figure()
    ppp=sns.kdeplot(data=data[orientation==c,1],bw=bw2,label="BW={:.3f}".format(bw2)) # THE PLot
    x,y = ppp.get_lines()[0].get_data()
    cdf = scipy.integrate.cumtrapz(y, x, initial=0)
    nearest_05 = np.abs(cdf-0.5).argmin()
    x_mean = x[nearest_05]
    y_mean = y[nearest_05]
    plt.vlines(x_mean, 0, y_mean)
    plt.title("Conditional Gaussian KDE of acc with orientation = {}".format(c))
    plt.show()
    print('Median acc value with orientation c={} is {:.5f}'.format(c,x_mean))
    print(np.mean(data[orientation==c,1]))
#Ref: https://stackoverflow.com/questions/28956622/how-to-locate-the-median-in-a-seaborn-kde-plot
    
#e. Joint KDE plots
for c in [2,3,4,5]:
    plt.figure()
    sns.kdeplot(data[orientation==c,0],data[orientation==c,1],cmap="Reds", shade=True, bw_method='silverman') #Silverman's bandwidth
    plt.title('Joint 2D Gaussian KDE contour for the orientation {}'.format(c))
    plt.xlabel('amygdala')
    plt.ylabel('acc')
    plt.show()
    
#%%
   
import scipy.io as spio
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ll
import numpy as np

mnsit_data=spio.loadmat('data/data.mat')
mnsit=mnsit_data['data']

mnsit_labels=spio.loadmat('data/label.mat')
label=mnsit_labels['trueLabel']
math.sqrt(mnsit.shape[0]) #28 pixels

fig = plt.figure()
fig.set_size_inches(2, 2)
ax = fig.add_subplot(111)
img = mnsit[:,1].reshape(28, 28).T
ax.imshow(img, aspect='auto', cmap=plt.cm.gray)


                    # PCA
m,n = mnsit.T.shape
mu = np.mean(mnsit.T,axis = 0) #Mean 
xc = mnsit.T - mu[None,:] #x-mu
C_pca = np.dot(xc,xc.T)/m   # Covariance Matrix
K = 4 #Number of principal components
S_pca,Wt_pca = ll.eigs(C_pca,k = K) #EIgen decomposition
S_pca = S_pca.real  # Top k Largest variances
Wt_pca = Wt_pca.real #Weights/Projection Directions
zpca = Wt_pca.dot(np.diag(S_pca**(-1/2)))
