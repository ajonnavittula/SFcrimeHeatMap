from pyexcel_xlsx import get_data
import numpy as np
import pandas as pd
import pyexcel as pe
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
data=pd.read_csv("/Users/Apoorva/Desktop/WPI/AI/Project/SF_Processed.csv",engine='python',delimiter=',', header=0)
#print("Total rows: {0}".format(len(data)))
d=data.as_matrix(columns=None)

temp = np.asarray(d)
#print temp.dtype
np.random.shuffle(temp)
temp2 = temp

pca = PCA(n_components=4)
pca.fit(temp[:,0:4])
#print pca.explained_variance_ratio_
# print pca.components_
X_train_n=temp2[0:1048575,0:4]
y_train_n=temp2[0:1048575,4]
X_train = temp2[0:629145,0:4]
X_train = np.array(X_train).astype(np.float)
#print X_train.dtype
#print ".........................."
y_train = temp2[0:629145,4]
y_train = np.array(y_train).astype(int)
print y_train
#print y_train.dtype
X_val = temp2[629145:838860,0:4]
X_val = np.array(X_val).astype(np.float)

y_val = temp2[629145:838860,4]
y_val = np.array(y_val).astype(np.int)
#print y_val.dtype
X_test = temp2[838860:1048575,0:4]
X_test = np.array(X_test).astype(np.float)

y_test = temp2[838860:1048575,4]
y_test = np.array(y_test).astype(np.float)
y_test = np.array(y_test).astype(np.int)
X_train_scaled=preprocessing.scale(X_train)
#print X_train_scaled.dtype
Y_train_scaled=preprocessing.scale(y_train).astype(np.int)
#print Y_train_scaled.dtype
X_val_scaled=preprocessing.scale(X_val)
y_val_scaled=preprocessing.scale(y_val)
X_test_scale=preprocessing.scale(X_test)
y_test_scale=preprocessing.scale(y_test)

skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X_train_n,y_train_n)
for train_index, test_index in skf.split(X_train_n,y_train_n):
	print("TRAIN:", train_index, "TEST:", test_index)
'''clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
clf.fit(X_train_scaled,Y_train_scaled)
predicion=clf.predict(X_test_scale)
print predicion
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#clf.fit(X_train_scaled,Y_train_scaled)
#print clf.predict(X_val_scaled)'''




