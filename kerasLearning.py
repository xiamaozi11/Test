# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:32:59 2018

@author: maojin.xia
"""

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import  Callback, ModelCheckpoint
import os

np.random.seed(1671) #重复性设置

#(X_train,y_train),(X_test,y_test) = mnist.load_data()


#mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

def load_data(path="mnist.npz"): 
    f = np.load(path) 
    x_train, y_train = f['x_train'], f['y_train']   
    x_test, y_test = f['x_test'], f['y_test']  
    f.close()    
    return (x_train, y_train), (x_test, y_test)


NB_EPOCH = 20
BATCH_SIZE = 128    
VERBOSE = 1
NB_CLASSES = 10 #输出类别数
OPTIMIZER = Adam() #SGD优化器
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 #训练集中用作验证集的数据比例
RESHAPED = 784
DROPOUT = 0.3
MODEL_DIR = "/tmp"

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.reshape(60000,RESHAPED)
x_test = x_test.reshape(10000,RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#归一化
x_train /= 255
x_test /=255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self,batch, logs={}):
        self.losses.append(logs.get('loss'))


#将类向量转化为二值类别矩阵,输出十个类别
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
#model.add(Dense(NB_CLASSES, input_shape = (RESHAPED,)))
#model.add(Activation('softmax'))

model.add(Dense(N_HIDDEN, input_shape = (RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
#model.add(Dense(64,input_dim =64, kernel_regularizer = regularizers.l2(0.01)))

model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer = OPTIMIZER, metrics=['accuracy'])

history = LossHistory()
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint= ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint(filepath)

model.fit(x_train,y_train, batch_size = BATCH_SIZE, epochs= NB_EPOCH, verbose= VERBOSE, validation_split= VALIDATION_SPLIT,callbacks = [checkpoint])
score = model.evaluate(x_test,y_test,verbose=VERBOSE)

print("Test score:", score[0])
print("Test accuracy:", score[1])
#print (history.losses)


