#!/usr/bin/env python
# -*- coding:utf-8 -*-
#       FileName:  predict.py
#
#    Description:
#
#        Version:  1.0
#        Created:  2019-10-08 16:50:42
#  Last Modified:  2019-10-09 17:02:27
#       Revision:  none
#       Compiler:  gcc
#
#         Author:  zt ()
#   Organization:

import tensorflow as tf
import random
import numpy as np
# import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model

cfg = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
cfg.gpu_options.per_process_gpu_memory_fraction = 0.9
cfg.allow_soft_placement = True
sess = tf.compat.v1.InteractiveSession(config=cfg)

# 数据集
# 划分MNIST训练集、测试集
(_, _), (X_test, y_test) = mnist.load_data()

# 随机数
index = random.randint(0, X_test.shape[0])
x = X_test[index]
y = y_test[index]

# 显示该数字
# plt.imshow(x, cmap='gray_r')
# plt.title("original {}".format(y))
# plt.show()

# 加载
mymodel = load_model('mnistmodel2.h5')

# 预测
x = x.reshape(1, 28, 28, 1)
predict = mymodel.predict(x)
# 取最大值的位置
predict = np.argmax(predict)

print('index', index)
print('original:', y)
print('predicted:', predict)
