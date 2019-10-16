#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  mnist.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-10-08 16:38:35
#  Last Modified:  2019-10-15 21:57:55
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense

# 数据集
# 读取并划分MNIST训练集、测试集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(len(X_train))

X_train = X_train.reshape(len(X_train), -1)  # 二维变一维
# print(X_train.shape)
X_test = X_test.reshape(len(X_test), -1)

X_train = X_train.astype('float32')  # 转为float类型
X_test = X_test.astype('float32')

X_train = (X_train - 127) / 127  # 灰度像素数据归一化
X_test = (X_test - 127) / 127

# 独热编码。如原来为5，转换后[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 定义模型
# Keras序列模型
model = Sequential()

# 添加全连接层（隐藏层），隐藏层数20层，激活函数为ReLU
model.add(Dense(20, input_shape=(784,), activation='relu'))
# 添加输出层，结果10类，激活函数为Sigmoid
model.add(Dense(10, activation='sigmoid'))

# 模型基本信息
print(model.summary())

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练
model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.025)  # 迭代20次

# 评估
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Accuracy:', accuracy)

# 保存
model.save('mnistmodel.h5')

