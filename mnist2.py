#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  mnist2.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-10-09 11:01:04
#  Last Modified:  2020-02-21 19:16:46
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import load_model

cfg = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
cfg.gpu_options.per_process_gpu_memory_fraction = 0.9
cfg.allow_soft_placement = True
sess = tf.compat.v1.InteractiveSession(config=cfg)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)  # 二维变一维
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype('float32')  # 转为float类型
X_test = X_test.astype('float32')

# 灰度像素数据归一化
X_train = (X_train - 127) / 127
X_test = (X_test - 127) / 127

# print(y_train.shape)
y_train = np_utils.to_categorical(y_train, num_classes=10)
# print(y_train.shape)
y_test = np_utils.to_categorical(y_test, num_classes=10)

'''
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=[28, 28, 1]))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(), metrics=['accuracy'])
'''

model = load_model('mnistmodel2.h5')

batch_size = 32
epochs = 10
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('loss:%.4f accuracy:%.4f' % (loss, accuracy))
model.save('mnistmodel2.h5')
