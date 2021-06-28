import os
import numpy as np
from os.path import abspath, exists
from scipy import sparse
import scipy
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import random


def output_file(a, idx2name, c_idx):
    dirpath = os.getcwd()
    node_file = dirpath + '//nodes.csv'
    edge_file = dirpath + '//edges.csv'

    with open(edge_file, 'w') as fid:
        fid.write('Source\tTarget\n')
        for i in range(len(a)):
            fid.write(f'{a[i,0]}\t{a[i,1]}\n')

    with open(node_file, 'w') as fid:
        fid.write('Id\tLabel\tColor\n')
        for i in range(len(idx2name)):
            fid.write(f'{i}\t{idx2name[i]}\t{c_idx[i]}\n')


def read_team_name():
    # read inverse_teams.txt file
    f_path = abspath("inverse_teams.txt")
    idx2name = []
    if exists(f_path):
        with open(f_path) as fid:
            for line in fid.readlines():
                name = line.split("\t", 1)[1]
                idx2name.append(name[:-1])
    return idx2name


def import_graph():
    # read the graph from 'play_graph.txt'
    f_path = abspath("play_graph.txt")
    if exists(f_path):
        with open(f_path) as graph_file:
            lines = [line.split() for line in graph_file]
    return np.array(lines).astype(int)



# spectral clustering
n = 321
k = 20

def spectral_clustering(n,k):
    # random.seed(400)
    # load the graph
    a = import_graph()
    
    i = a[:, 0]-1
    j = a[:, 1]-1
    v = np.ones((a.shape[0], 1)).flatten()
    
    A = sparse.coo_matrix((v, (i, j)), shape=(n, n))
    A = (A + np.transpose(A))/2
    A = sparse.csc_matrix.todense(A) # ## convert to dense matrix
    
    D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1)
    L = D @ A @ D
    L = np.array(L) # ## covert to array
    
    # v, x = np.linalg.eig(L)
    # x = x[:, 0:k].real
    
    # eigendecompoosition
    v, x= np.linalg.eig(L)
    
    plt.plot(np.sort(v)[::-1]) # Plot of Eigen values from largest to smallest 
    plt.title('2.2 Spectrum plot')
    plt.xlabel('node')
    plt.ylabel('Eigen values')
    plt.savefig('spectrum plot.png')
    
    
    idx_sorted = np.argsort(v) # the index of eigenvalue sorted acsending
    
    x = x[:, idx_sorted[-k:]] # select the k largest eigenvectors
    #NOte I chose largest K values because of using the 'Normalized Cuts' variant
    #(https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf)
    
    x = x/np.repeat(np.sqrt(np.sum(x*x, axis=1).reshape(-1, 1)), k, axis=1)
    
    # scatter
    # plt.scatter(x[:, 0], x[:, 1])
    # plt.show()
    
    
    # k-means
    kmeans = KMeans(n_clusters=k).fit(x.real)
    c_idx = kmeans.labels_
    import warnings
    warnings.filterwarnings("ignore")
    
    #Number of teams in each cluster
    (unique, counts) = np.unique(c_idx, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
   # print("\n Number of teams in each of the %d" %k +" Clusters")
    print ("Iteration ")
    print(np.sort(frequencies[:,1]))
    
    # show cluster
    idx2name = read_team_name()
    for i in range(k):
        print(f'Cluster {i+1}\n***************')
        idx = [index for index, t in enumerate(c_idx) if t == i]
        for index in idx:
            print(idx2name[index])
        print('\n')
    
    # output file
    output_file(a, idx2name, c_idx)


for k in [5,7,10,20]:
    spectral_clustering(n, k)
    
