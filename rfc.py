from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import FastICA, PCA
import csv     # imports the csv module
import sys      # imports the sys module
import numpy as np
import matplotlib.pyplot as plt



f = open("SF_ProcessedScaled.csv", "rb") # opens the csv file
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

clf2 = RandomForestClassifier(max_depth=16,n_estimators=1024)
print "Training..."
clf2.fit(Xtr, y_train)
y_train_pred = clf2.predict(Xtr)
print "Training Complete..."
print "Testing..."
accutr = np.mean(y_train == y_train_pred)                         
y_test_pred = clf2.predict(Xts)
accut = np.mean(y_test == y_test_pred)

print "Train Accuracy = %f Test Accuracy = %f "%(accutr,accut)

y_train_pred = y_train_pred.T
print "Testing..."
accutr = np.mean(y_train == y_train_pred)                         

y_test_pred = y_test_pred.T
accut = np.mean(y_test == y_test_pred)

print "Train Accuracy = %f Test Accuracy = %f "%(accutr,accut)


