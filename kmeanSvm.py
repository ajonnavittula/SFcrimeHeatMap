# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 15:20:58 2016

@author: krish
"""
#Kmeans and SVM clustering


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd

from sklearn.cluster import MiniBatchKMeans


data=pd.read_csv("SF_ProcessedEightFeatures.csv",engine='python',delimiter=',', header=0)
d=data.as_matrix(columns=None)
temp = np.asarray(d)
np.random.shuffle(temp)
temp2 = temp

pca = PCA(n_components=8)
pca.fit(temp[:,0:8])

X_train_n=temp2[0:1048575,0:8]
y_train_n=temp2[0:1048575,8]
X_train = temp2[0:629145,0:8]
X_train = np.array(X_train).astype(np.float)

y_train = temp2[0:629145,8]
y_train = np.array(y_train).astype(int)
#print(y_train)
#print y_train.dtype
X_val = temp2[629145:838860,0:8]
X_val = np.array(X_val).astype(np.float)

y_val = temp2[629145:838860,8]
y_val = np.array(y_val).astype(np.int)
#print y_val.dtype
X_test = temp2[838860:1048575,0:8]
X_test = np.array(X_test).astype(np.float)

y_test = temp2[838860:1048575,8]
y_test = np.array(y_test).astype(np.float)
y_test = np.array(y_test).astype(np.int)

#perform kmeans here
kmeans = MiniBatchKMeans(n_clusters=10000,init='random')
kmeans.fit(X_train)