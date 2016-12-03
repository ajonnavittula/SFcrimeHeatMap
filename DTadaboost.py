from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import measure
import time

import pandas as pd

#data=pd.read_csv("SF_ProcessedEightFeatures.csv",engine='python',delimiter=',', header=0)
data=pd.read_csv("SF_unique_all.csv",engine='python',delimiter=',', header=0)
d=data.as_matrix(columns=None)

#X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
#                               n_classes=3, random_state=1)

#n_split = 3000

#X_train, X_test = X[:n_split], X[n_split:]
#y_train, y_test = y[:n_split], y[n_split:]

temp = np.asarray(d)
#print temp.dtype
np.random.shuffle(temp)
temp2 = temp

pca = PCA(n_components=6)
pca.fit(temp[:,0:6])
#print pca.explained_variance_ratio_
# print pca.components_
#X_train_n=temp2[0:1048575,0:8]
#y_train_n=temp2[0:1048575,8]
#X_train = temp2[0:629145,0:8]
#X_train = np.array(X_train).astype(np.float)
##print X_train.dtype
#print ".........................."
#y_train = temp2[0:629145,8]
#y_train = np.array(y_train).astype(int)
##print(y_train)
##print y_train.dtype
#X_val = temp2[629145:838860,0:8]
#X_val = np.array(X_val).astype(np.float)
#
#y_val = temp2[629145:838860,8]
#y_val = np.array(y_val).astype(np.int)
##print y_val.dtype
#X_test = temp2[838860:1048575,0:8]
#X_test = np.array(X_test).astype(np.float)
#
#y_test = temp2[838860:1048575,8]
#y_test = np.array(y_test).astype(np.float)
#y_test = np.array(y_test).astype(np.int)
#
#print y_train
performanceArray = np.zeros([5,4])
for iter in range(0,5):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(temp[:,2:4],temp[:,6],test_size=0.20,stratify=temp[:,6])
    bdt_discrete = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        learning_rate=0.5,
        algorithm="SAMME")
    
    bdt_discrete.fit(X_train, y_train)
    timetaken = time.time() - start_time
    performanceArray[iter,0] = timetaken
    
    ##########################################################################################
    #clf2.fit(Xtr, y_train)
    y_train_pred = bdt_discrete.predict(X_train)
    print "Training Complete..."
    print "Testing..."
    accutr = np.mean(y_train == y_train_pred)                         
    y_test_pred = bdt_discrete.predict(X_test)
    accut = np.mean(y_test == y_test_pred)
    
    print "Train Accuracy = %f Test Accuracy = %f "%(accutr,accut)
    
    y_train_pred = y_train_pred.T
    print "Testing..."
    accutr = np.mean(y_train == y_train_pred)                         
    
    y_test_pred = y_test_pred.T
    accut = np.mean(y_test == y_test_pred)
    
    print "Train Accuracy = %f Test Accuracy = %f "%(accutr,accut)
    
    ################################################################################################
    performanceArray[iter,1:] = measure.measurements(y_test_pred,y_test)


