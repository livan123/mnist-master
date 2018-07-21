#!/usr/bin/env python
# _*_ UTF-8 _*_

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

data_sets = input_data.read_data_sets('E:/Python_workspace/mnist-master/Mnist_data/')
images = data_sets.train.images
labels = data_sets.train.labels

total = images.shape[0]
print(images.shape)
print(images)
im = images[7]
im2 = np.array(im)
print(im)
im2 = im2.reshape(28,28)
print(im2)
fig = plt.figure()
plotwindow = fig.add_subplot(1,1,1)
plt.imshow(im2, cmap='gray')
plt.show()
