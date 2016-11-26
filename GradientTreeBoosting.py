from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import FastICA, PCA
import csv     # imports the csv module
import sys      # imports the sys module
import numpy as np
import matplotlib.pyplot as plt



f = open("SF_ProcessedEightFeatures.csv", "rb") # opens the csv file
i = 0
r = []
try:
    reader = csv.reader(f)  # creates the reader object
    for row in reader:   # iterates the rows of the file in orders
        #print row    # prints each row
        r.append(row)
finally:
    f.close()      # closing

####################################
temp = np.asarray(r)

num_train = 629145
num_val = 209715
num_test = 209715

X_train = temp[0:629145,0:4]
X_train = np.array(X_train).astype(np.float)

y_train = temp[0:629145,4]
y_train = np.array(y_train).astype(np.float)

X_val = temp[629145:838860,0:4]
X_val = np.array(X_val).astype(np.float)

y_val = temp[629145:838860,4]
y_val = np.array(y_val).astype(np.float)

X_test = temp[838860:1048575,0:4]
X_test = np.array(X_test).astype(np.float)

y_test = temp[838860:1048575,4]
y_test = np.array(y_test).astype(np.float)

print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape


# For comparison, compute PCA
pca = PCA(n_components=4)
Xtr = pca.fit_transform(X_train)  # Reconstruct signals based on orthogonal components
Xts = pca.fit_transform(X_test)  # Reconstruct signals based on orthogonal components

#clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0) Train Accuracy = 0.921114 Test Accuracy = 0.878907 
#clf2 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,max_depth=3, random_state=0) Train Accuracy = 0.983857 Test Accuracy = 0.879990 
#clf2 = GradientBoostingClassifier(n_estimators=400, learning_rate=0.02,max_depth=3, random_state=0) #Train Accuracy = 0.978574 Test Accuracy = 0.882669

clf2 = GradientBoostingClassifier(n_estimators=800, learning_rate=0.4,max_depth=3, random_state=0) #Train Accuracy = 0.979995 Test Accuracy = 0.877224 

clf2.fit(Xtr, y_train)
accut=clf2.score(Xts,y_test)
accutr=clf2.score(Xtr,y_train)
print "-------------------------------------Train Accuracy = %f Test Accuracy = %f "%(accutr,accut)


