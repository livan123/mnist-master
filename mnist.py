#!/usr/bin/env python
# _*_ UTF-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensorflow.python.ops.logging_ops import *
import tensorflow.python.platform
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# 总的流程为：
# 建模——计算损失——训练——评估：

# 建模
def inference(images, hidden1_units, hidden2_units):
  """
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: 第一层的大小.
    hidden2_units: 第二层的大小.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    # 截尾正态
    weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
    # matmul：矩阵相乘
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

# 计算损失值：
def loss(logits, labels):
  batch_size = tf.size(labels)
#   tf.expand_dims：增加一个纬度；
  labels = tf.expand_dims(labels, 1)
#   tf.range:创建数字序列
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
#   tf.concat:连接两个矩阵
  concated = tf.concat(1, [indices, labels])
#   稀疏矩阵转密集矩阵
  onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
#   计算logits经softmax函数激活之后的交叉熵；
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,onehot_labels,name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

# 训练
def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  scalar_summary(loss.op.name, loss)
  # 创建一个实现梯度下降算法的优化器；
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

# 评估
def evaluation(logits, labels):
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
#   in_top_k：用于计算预测的结果与实际结果是否相等；
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
#   cast：类型转化函数
  return tf.reduce_sum(tf.cast(correct, tf.int32))
