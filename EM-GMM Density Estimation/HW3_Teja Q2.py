import scipy.io as spio
import numpy as np
import numpy.matlib
import pandas as pd
import seaborn as sns
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans

mnsit_data=spio.loadmat('data/data.mat')
mnsit=mnsit_data['data']

mnsit_labels=spio.loadmat('data/label.mat')
label=mnsit_labels['trueLabel']
math.sqrt(mnsit.shape[0]) #28 pixels

fig = plt.figure()
fig.set_size_inches(2, 2)
ax = fig.add_subplot(111)
img = mnsit[:,11].reshape(28, 28).T
ax.imshow(img, aspect='auto', cmap=plt.cm.gray)

                    # PCA
m,n = mnsit.T.shape
mu = np.mean(mnsit.T,axis = 0) #Mean 
xc = mnsit.T - mu[None,:] #x-mu
C_pca = np.dot(xc,xc.T)/m   # Covariance Matrix
d = 4 #Number of principal components
S_pca,Wt_pca = ll.eigs(C_pca,k = d) #EIgen decomposition
S_pca = S_pca.real  # Top k Largest variances
Wt_pca = Wt_pca.real #Weights/Projection Directions
pdata = Wt_pca.dot(np.diag(S_pca**(-1/2)))
z= np.dot(Wt_pca.T,xc)
#z= np.dot(Wt_pca.T,xc)/math.sqrt(S_pca[0])
#pdata = np.dot(mnsit.T,z.T)
 
##INITIALISING

# EM-GMM for data
# number of mixtures
K = 2

# random seed
seed = 400

# initialize prior - PI
np.random.seed(seed)
pi = np.random.random(K)
pi = pi/np.sum(pi)

# initial mean and covariance - MU & SIGMA
np.random.seed(seed)
mu = np.random.randn(K,d) #sample (or samples) from the “standard normal” distribution 
mu_old = mu.copy()

sigma = []
for ii in range(K):
    # to ensure the covariance psd
    # np.random.seed(seed)
    s = np.random.randn(d, d)
    sigma.append(s@s.T)

## Performing transofrmation on sigma and adding to identity matrix
S1 = np.array(sigma[0])
S2 = np.array(sigma[1])
sigma1 = S1.dot(S1.T) + np.identity(d)
sigma2 = S2.dot(S2.T) + np.identity(d)

sigma = [sigma1,sigma2]
    
# initialize the posterior - TAU
tau = np.full((m, K), fill_value=0.)

# # parameter for countour plot
# xrange = np.arange(-5, -5, 0.1)
# yrange = np.arange(-5, -5, 0.1)
# ####
maxIter= 100
tol = 1e-9

plt.ion()

#%%
loglikelihood=[]
temppiN=tau.copy()
plot = defaultdict(list)

for ii in range(100):

    # E-step    
    for kk in range(K):
        tau[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
    # normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (m,1)    
    tau = np.divide(tau, np.tile(sum_tau, (1, K)))    
   
    # M-step
    for kk in range(K):
        # update prior
        pi[kk] = np.sum(tau[:, kk])/m
        
        # update component mean
        mu[kk] = pdata.T @ tau[:,kk] / np.sum(tau[:,kk], axis = 0)
        
        # update cov matrix
        dummy = pdata - np.tile(mu[kk], (m,1)) # X-mu
        sigma[kk] = dummy.T @ np.diag(tau[:,kk]) @ dummy / np.sum(tau[:,kk], axis = 0)
        temppiN[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])

    loglikelihood.append(sum(np.log(np.sum(temppiN, axis=1))))
    
    print(ii,np.linalg.norm(mu-mu_old))    
    if np.linalg.norm(mu-mu_old) < tol:
        print('training coverged')
        break
    mu_old = mu.copy()
    if ii==99:
        print('max iteration reached')
        break
    
## Log-likelihood plot    
plt.figure()
plt.plot(loglikelihood)
plt.title('log likelihood graph is converging')
plt.xlabel('iteration')
plt.ylabel('Log-likelihood')
plt.show()
     
    
#%%    
#GMM Gaussian model 
print("Weights of each component is {}".format(pi))
print("GMM MOdel with Mean: {}".format(mu))
# print("GMM MOdel with Variance : {}".format(sigma))

#Mean images of the gaussians
plt.imshow((z.T.dot(mu[0])+ np.mean(mnsit, axis = 1)).reshape(28,28,order="F"), cmap='gray')
plt.title('Component 1')
plt.show()
plt.imshow((z.T.dot(mu[1])+ np.mean(mnsit, axis = 1)).reshape(28,28,order="F"), cmap='gray')
plt.title('Component 2')
plt.show()

#Heatmap of the covariance mattix
sns.heatmap(sigma[0], annot=True)
plt.title("Heatmap of Covariance Matrix 1")
plt.show()
sns.heatmap(sigma[1], annot=True)
plt.title("Heatmap of Covariance Matrix 2")
plt.show()


#%%
#Weights of Gaussians ---- tau 

true_labels = spio.loadmat('data/label.mat')['trueLabel'].flatten()
predicted_labels = np.argmax(tau, axis=1)
pd.crosstab(true_labels, predicted_labels)

#Misclassification of 2's as 6's = ~ 6.5%
67/(67+965)*100
#Misclassification of 6's as 2's = ~ 0.9%
9/(9+949)*100


## Clustering using KMeans
km = KMeans(n_clusters=2, init='random',n_init=10, max_iter=300,tol=1e-04, random_state=0)
kmeans = km.fit_predict(np.dot(mnsit.T,z.T))
pd.crosstab(true_labels, kmeans)

#Misclassification of 2's as 6's  = ~ 6.3%
65/(65+967)*100
#Misclassification of 6's as 2's  = ~ 8.55%
82/(82+876)*100
