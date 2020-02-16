"""
Created on Tue Jan 7 2020
@author: sumansahoo16
Private Score : 0.99100
Public Score : 0.98900
"""

import pandas as pd
import numpy as np
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau


X_train = pd.read_csv('train.csv')
y_train = X_train['label']
X_train.drop('label',axis=1,inplace=True)

test = pd.read_csv('test.csv')
test.drop('id',axis=1,inplace=True)

X_test = pd.read_csv('Dig-MNIST.csv')
y_test = X_test['label']
X_test.drop('label',axis=1,inplace=True)

y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)

X_train = X_train / 255.0
X_test = X_test / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.3, height_shift_range = 0.3, shear_range = 0.15, zoom_range = 0.3)
datagen.fit(X_train)

#################################################################################################################
model = Sequential()
model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
#################################################################################################################

epochs = 70
batch_size = 1024

checkpoint = ModelCheckpoint("best_weights.hdf5", monitor='val_accuracy', save_best_only=True, mode='max')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.7, min_lr=0.00001)

optimizer = RMSprop(lr = 0.001)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


history = model.fit(datagen.flow(X_train,y_train, batch_size=batch_size), epochs = epochs, validation_data=(X_test,y_test), callbacks=[checkpoint,learning_rate_reduction])

model.load_weights("best_weights.hdf5")

results = model.predict(test)
results = np.argmax(results,axis = 1)

submission = pd.read_csv('sample_submission.csv')
submission['label'] = results
submission.to_csv("submission.csv",index=False)
