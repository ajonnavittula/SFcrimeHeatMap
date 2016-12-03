#agglo clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("SF_Processed.csv",engine='python',delimiter=',',header=0)
d=data.as_matrix(columns=None)

data_set = np.array([d])
np.random.shuffle(data_set)

X_train = data_set[0:1048575,0:4]
y_train = data_set[0:1048575,4]

X_train = data_set[838860:1048575,0:4]
y_train = data_set[838860:1048575,4]

# X = StandardScaler().fit_transform(X_train)
X = X_train


agglo = cluster.FeatureAgglomeration(n_clusters=6)

agglo.fit(X)
X_reduced = agglo.transform(X)

X_restored = agglo.inverse_transform(X_reduced)
# images_restored = np.reshape(X_restored, images.shape)
plt.figure(1, figsize=(4, 3.5))