import csv     # imports the csv module
import sys      # imports the sys module
import numpy as np
from sklearn.decomposition import PCA
#from sklearn import datasets

f = open("SF_Processed.csv", "rb") # opens the csv file
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

np.random.shuffle(temp)
temp2 = temp
num_train = 629145
num_val = 209715
num_test = 209715
    
X_train = temp2[0:629145,0:4]
X_train = np.array(X_train).astype(np.float)

y_train = temp2[0:629145,4]
y_train = np.array(y_train).astype(np.float)

X_val = temp2[629145:838860,0:4]
X_val = np.array(X_val).astype(np.float)

y_val = temp2[629145:838860,4]
y_val = np.array(y_val).astype(np.float)

X_test = temp2[838860:1048575,0:4]
X_test = np.array(X_test).astype(np.float)

y_test = temp2[838860:1048575,4]
y_test = np.array(y_test).astype(np.float)
y_test = np.array(y_test).astype(np.int)

print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

pca = PCA(n_components=4)
pca.fit(temp[:,0:4])
pca_score = pca.explained_variance_ratio_
V = pca.components_