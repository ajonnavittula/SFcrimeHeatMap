# import keras
import csv
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from keras.layers.core import Dense
# from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
# from keras.utils.np_utils import to_categorical
# from keras.callbacks import CSVLogger
# from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
# from keras.callbacks import LearningRateScheduler
# from keras.callbacks import ModelCheckpoint
import h5py
from sklearn.neural_network import MLPClassifier

np.random.seed(7)

def schedule(epoch):
	if epoch > 4:
		return 0.0001
	else:
		return 0.001

data=pd.read_csv("SF_ProcessedEightFeatures.csv",engine='python',delimiter=',', header=0)
#print("Total rows: {0}".format(len(data)))
d=data.as_matrix(columns=None)

temp = np.asarray(d)
print temp.shape
X_train, X_test, y_train, y_test = train_test_split(temp[:,:8],temp[:,8],test_size=0.20,stratify=temp[:,8])
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

X_train = scale(X_train)
X_test = scale(X_test)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 300), random_state=1)
clf.fit(X_train, y_train) 
outputs = clf.predict(X_test)

acc = 1 - np.count_nonzero(outputs - y_test)/len(y_test)
# acc = clf.score(X_test,y_test)

# model = Sequential()
# model.add(Dense(30,input_shape=(8,),activation='relu'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(6,activation='softmax'))

# sgd_obj = SGD(lr = 0.001)
# Adam_obj = Adam(lr = 0.0001)
# model.compile(optimizer=Adam_obj,loss='categorical_crossentropy',metrics=['accuracy'])

# csv = CSVLogger('training_log.csv')
# lrsched = LearningRateScheduler(schedule)
# checkpoint = ModelCheckpoint('model_weights.hdf5',verbose=1,monitor='val_acc',save_best_only=True)

# history = model.fit(X_train,y_train,nb_epoch=100, verbose=1, validation_data=(X_test,y_test), callbacks=[csv,checkpoint])
# trg_acc = history.history['acc']
# test_acc = history.history['val_acc']

# plt.plot(trg_acc)
# plt.plot(test_acc)
# plt.title('Accuracy vs Epochs')
# plt.legend(['Training Accuracy','Test Accuracy'])
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy %')
# plt.show()
