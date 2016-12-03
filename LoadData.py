#from sklearn.externals.six.moves import zip

#import matplotlib.pyplot as plt

import numpy as np

#from sklearn.datasets import make_gaussian_quantiles
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.decomposition import PCA

import pandas as pd
def LoadData():
	data=pd.read_csv("SF_ProcessedEightFeatures.csv",engine='python',delimiter=',', header=0)
	d=data.as_matrix(columns=None)

	temp2 = np.asarray(d)
	#print temp.dtype
	np.random.shuffle(temp2)
	temp = temp2

	num_train = 629145
	num_val = 209715
	num_test = 209715
	
	X_train = temp[0:838860,0:8]
	#X_train = np.array(X_train).astype(np.float)

	y_train = temp[0:838860,8]
	#y_train = np.array(y_train).astype(np.float)

	#X_val = temp[629145:838860,0:8]
	#X_val = np.array(X_val).astype(np.float)

	#y_val = temp[629145:838860,8]
	#y_val = np.array(y_val).astype(np.float)

	X_test = temp[838860:1048575,0:8]
	#X_test = np.array(X_test).astype(np.float)

	y_test = temp[838860:1048575,8]
	#y_test = np.array(y_test).astype(np.float)

	print 'Training data shape: ', X_train.shape
	print 'Training labels shape: ', y_train.shape
	print 'Test data shape: ', X_test.shape
	print 'Test labels shape: ', y_test.shape

	return X_train,y_train,X_test,y_test
