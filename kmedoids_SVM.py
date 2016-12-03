import random
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import k_medoids

def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in xrange(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C

data=pd.read_csv("SF_ProcessedEightFeatures.csv",engine='python',delimiter=',', header=0)
d=data.as_matrix(columns=None)

temp = np.asarray(d)
print temp.shape
X_train, X_test, y_train, y_test = train_test_split(temp[:,:8],temp[:,8],test_size=0.20,stratify=temp[:,8])
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

X_train = scale(X_train)
X_test = scale(X_test)

N_medoids = kMedoids(X_train,10000)
n_meds = k_medoids

