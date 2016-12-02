from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import numpy as np

a= np.zeros(3)
def measurements(y_true,y_pred):
	a[0]=f1_score(y_true, y_pred,average='weighted')
	a[1]=accuracy_score(y_true, y_pred)
	a[2]=precision_score(y_true,y_pred,average='weighted')
	
	return a

