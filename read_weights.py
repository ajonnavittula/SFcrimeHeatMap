import keras
import csv
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import h5py

np.random.seed(7)

data=pd.read_csv("SF_ProcessedEightFeatures.csv",engine='python',delimiter=',', header=0)
#print("Total rows: {0}".format(len(data)))
d=data.as_matrix(columns=None)

temp = np.asarray(d)

X_train, X_test, y_train, y_test = train_test_split(temp[:,:8],temp[:,8],test_size=0.20,stratify=temp[:,8])
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

X_train = scale(X_train)
X_test = scale(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(30,input_shape=(8,),activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(6,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights('model_weights.hdf5')

metrics = model.evaluate(X_test,y_test,verbose=0)

print metrics[1]*100