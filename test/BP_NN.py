#!/usr/bin/env python
# _*_ UTF-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
from cv2 import *

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
# 输入图片：
def imageprepare():
    file_name='E:/Python_workspace/mnist-master/Mnist_data/7.png'
    im = Image.open(file_name).convert('L')
    tv = list(im.getdata()) 
    tva = [(255-x)*1.0/255.0 for x in tv] 
    return tva
result=imageprepare()
print(result)
# 输入训练集
mnist = input_data.read_data_sets('E:/Python_workspace/mnist-master/Mnist_data/', one_hot=True)
# 需要多少层、每层有多少个节点，多个案例循环处理，得到多组分类，然后多个结果使用混淆矩阵，判断哪个的效果比较好；
# 模型构建：
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
a = tf.nn.softmax(tf.matmul(x, w)+b)
# 模型调参：
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(a), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

# 开始训练：
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    train.run({x:batch_xs, y:batch_ys})
# prediction在此时为训练好的模型，argmax是为了获取到a的最大概率所在的下标值，并将下标值作为判断的数值传给prediction；
prediction=tf.argmax(a,1)
print(prediction)
predint=prediction.eval(feed_dict={x:[result],keep_prob:1.0}, session=sess)
print(predint)

# 参数的循环组合；
# 输入的数据；
# 训练的数据；
# 建立的模型；
# 开始训练；
# 开始预测；
# 混淆矩阵；

