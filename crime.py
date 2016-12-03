
import pandas as pd
import numpy as np

import matplotlib.pylab as pylab
pylab.style.use('ggplot')
from LoadData import LoadData

(X_train,y_train,X_test,y_test) = LoadData()
counts = X_train[:,0:5]
np.asarray(counts, dtype=float, order=None)
# %matplotlib inline
counts.value_counts().plot('bar', logy=True)
counts.value_counts()
