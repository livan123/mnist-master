#!/usr/bin/env python
# _*_ UTF-8 _*_
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.variable_scope import get_variable
from tensorflow.python.framework import ops

# 1、矩阵的生成：
# sess = tf.InteractiveSession()
# tensor = [[1,2,3],[4,5,6]]
# x = tf.zeros([4,4], int32)
# y = tf.zeros_like(tensor)
# z = tf.fill([2,3], 2)
# z1 = tf.constant([1,2,3],shape=[3,2])
# z2 = tf.truncated_normal(shape=[1,5], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# w = tf.Variable(tf.zeros([3,3]))
# # 在一定的共享空间中使用，配合reuse与tf.variable_scope()：
# z3 = tf.get_variable("w", shape=[7,4], initializer=tf.contrib.layers.xavier_initializer())
# print(sess.run(z3))

# 2、矩阵的变化：
# sess = tf.InteractiveSession()
# labels = [1,2,3]
# mark1 = [4,5,6]
# mark2 = [4,5,6]
# a = [labels, mark1, mark2]
# b = [labels, mark1, mark2]
# # 定位矩阵的形状
# shape = tf.shape(labels)
# # 增加一个维度；[]--->[[],[]]
# ss = tf.expand_dims(labels, 0)
# # 将同一纬度融合成一个，只做融合，不做维度扩展，如果一维变两维，需要配合expand_dims使用；
# tt = tf.concat([a,b], axis = 0)
# # 按照第一维随机排序
# ff = tf.random_shuffle(a)
# # 最大值/最小值的下标
# rr1 = tf.argmax(labels)
# rr2 = tf.argmin(labels)
# # 判断两个张量是否相同
# qq = tf.equal(a, b)
# # 将矩阵做好类型转换
# aa = tf.cast(qq, dtype=int32)
# # 矩阵乘法
# vv = tf.matmul(a, b)
# # 矩阵变形
# re = tf.reshape(tt, (3,6))
# print(sess.run(re))
# sess = tf.InteractiveSession()
# # 生成等差序列
# x = tf.linspace(start = 1.0, stop = 5.0, num = 5)
# y = tf.range(start=1, limit=5, delta=1)
# # 更新模型中变量的值：实现a=b;
# a = tf.Variable(0.0)
# b = tf.placeholder(dtype=tf.float32, shape=[])
# op = tf.assign(a, b)
# sess.run(tf.initialize_all_variables())
# print(sess.run(a))
# sess.run(op, feed_dict={b:5.})
# print(sess.run(a))

# 3、命名空间：
# tf.variable_scope("foo"):空间名称不同，定义的变量互补干扰，使用get_variable()可以共享变量；
# 如果在相同命名空间下，且不可重用：不存在的变量会被新建；
# 如果在相同命名空间下，且可以重用：存在的变量会被返回；
# ops.reset_default_graph()
# sess = tf.InteractiveSession()
# with tf.variable_scope("scope1"):
#     w1 = tf.get_variable("w1", initializer=4.)
#     w2 = tf.Variable(0.0, name="w2")
# with tf.variable_scope("scope2"):
#     w1_p = tf.get_variable("w1", initializer=5.)
#     w2_p = tf.Variable(1.0, name="w2")
# with tf.variable_scope("scope1", reuse=True):
#     w1_reuse = tf.get_variable("w1")
#     w2_reuse = tf.Variable(2.0, name="w2")
# def compare_var(var1, var2):
#     print("----------------------------")
#     if var1 is var2:
#         print(sess.run(var2))
#     print(var1.name, var2.name)
# sess.run(tf.global_variables_initializer())
# print(sess.run(w1))
# print(sess.run(w2))
# print(sess.run(w1_p))
# print(sess.run(w2_p))
# print(sess.run(w1_reuse))
# print(sess.run(w2_reuse))
# compare_var(w1, w1_p)
# compare_var(w2, w2_p)
# compare_var(w1, w1_reuse)
# compare_var(w2, w2_reuse)

# 4、计算图的使用
# sess = tf.InteractiveSession()
# log_dir = 'E:/Python_workspace/pic'
# def practice_num():
#     with tf.name_scope("input1"):
#         input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
#     with tf.name_scope("input2"):
#         input2 = tf.Variable(tf.random_uniform([3]),name="input2")
#     output = tf.add_n([input1, input2], name="add")
#     sess.run(tf.global_variables_initializer())
#     sess.run(output)
#     writer = tf.summary.FileWriter(log_dir+"/log", tf.get_default_graph())
#     writer.close()
# practice_num()

# 5、神经网络相关函数：
# # 选取tensor中索引对应的元素：
# c = np.random.random([10, 1])
# # 取出第二个、第四个元素
# b = tf.nn.embedding_lookup(c, [1,3])
# sess = tf.Session()
# print(sess.run(b))
# print(c)
#  
# a = tf.get_variable('a', shape=[5, 2])
# b = tf.get_variable('b', shape=[5, 2], trainable=False)
# # 获取可以训练的变量a；
# tvar = tf.trainable_variables()
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# print(sess.run(tvar))
# print(sess.run(a))
# 
# 神经网络梯度规则：
# 1）梯度求导：
# grad = tf.gradients(ys=b, xs=a):即对b进行求导，自变量为a；
# grad = tf.gradients(ys=[y1, y2], xs=[x1,x2,x3]):则相当于对y1+y2求偏导,偏导的自变量为[x1, x2, x3]；
# x_input = tf.placeholder(tf.float32, name='x_input')
# y_input = tf.placeholder(tf.float32, name='y_input')
# w = tf.Variable(2.0, name='weight')
# b = tf.Variable(1.0, name='biases')
# y = tf.add(tf.multiply(x_input, w), b)
# gradient:梯度求导；
# loss = tf.reduce_mean(tf.pow(y_input - y, 2))/(2*32)
# train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# # train为训练之后的模型
# batch_xs, batch_ys = mnist.train.next_batch(100)
# train.run({x:batch_xs, y:batch_ys})
# print(sess.run(y, feed_dict={x_input:x_train[i], y_input:y_train[i]}))
# 2）加权重求导：即对xs中的每个元素的求导值加权重。
# w1 = tf.get_variable('w1', shape=[1])
# w2 = tf.get_variable('w2', shape=[1])
# w3 = tf.get_variable('w3', shape=[1])
# w4 = tf.get_variable('w4', shape=[1])
# z1 = 3*w1+2*w2+w3
# z2 = -1*w3+w4
# # , grad_ys=[[-2.0, -3.0, -4.0],[-2.0, -3.0, -4.0]]
# # 原来的值是对（z1+z2）求偏导；
# # 现在的值是对（(-2)*z1+(-3)*z2）求偏导；
# grads = tf.gradients([z1, z2], [w1, w2, w3, w4], grad_ys=[[-2.0],[-3.0]])
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(grads))
# 3）梯度阻挡：
# 主要是阻挡一个方向的梯度，使其不再向上传播：
# w1 = tf.Variable(2.0)
# w2 = tf.Variable(2.0)
# a = tf.multiply(w1, 3.0)
# a_stoped = tf.stop_gradient(a)
# b = tf.multiply(a_stoped, w2)
# # b = w1*3.0*w2
# gradients = tf.gradients(b, xs=[w1, w2])
# # 上面的模型是为了构建一个含有两个变量的函数式，对b求导时主要分两步：
# # 第一步求w1的导数，因为在运算中针对w1方向的梯度已经停掉，所以输出none；
# # 第二步求w2的导数，运算中针对w2方向的梯度正常运算，所以输出正常的值；
# # 可以将b函数看作是一个树状结构，stop_gradient限制了一个分枝，其他分枝不受影响；
# print(gradients)
# 4）梯度修剪（梯度爆炸控制：由于权重的更新过于迅猛）：
# 梯度爆炸与梯度消减的原因一样，都是因为链式法则求导的关系，导致梯度指数级衰减，为了避免梯度爆炸，需要进行梯度修剪；
# tf.clip_by_global_norm(t_list, clip_norm, use_norm, name):
# gradients得到的是一个梯度，有时这个梯度太大，导致梯度爆炸，因此需要进行梯度的修剪，
# 主要是按照clip_norm的平方和对t_list进行修剪，得到修剪后的梯度值。
# 原理为：
# 第一步：在solver中先设置一个clip_gradient;
# 第二步：在进行前向传播或者后向传播时，会得到每个权重的梯度diff，这时不像通常那样直接使用这些梯度进行权重更新，而是先求所有权重梯度的平方和sumsq_diff，
# 缩放因子为：scale_factor=clip_gradient/sumsq_diff;这个scale_factor在（0,1）之间，如果权重梯度的平方和越大，则缩放因子将越小；
# 第三步：将所有的权重梯度乘以这个缩放因子，这时得到的梯度才是最后的梯度信息；
# import tensorflow as tf  
# def gradient_clip(gradients, max_gradient_norm):
#     """计算的过程为：clipped_gradients=gradients/平方和(max_gradient_norm)"""
#     clipped_gradients, gradient_norm = tf.clip_by_global_norm(
#             gradients, max_gradient_norm)
#     gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
#     gradient_norm_summary.append(
#         tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
#     return clipped_gradients
# w1 = tf.Variable([[3.0,2.0]])  
# params = tf.trainable_variables()
# res = tf.matmul(w1, [[3.0],[1.]])  
# grads = tf.gradients(res,[w1])  
# clipped_gradients = gradient_clip(grads,2.0)
# global_step = tf.Variable(0, name='global_step', trainable=False)
# with tf.Session() as sess:  
#     tf.global_variables_initializer().run()
#     print(sess.run(res))
#     print(sess.run(grads))  
#     print(sess.run(clipped_gradients))
# dropout(x, keep_drop)函数：主要是防止训练过程中的过拟合，按照概率将x中的元素置为零，并将其他值放大；
# x是一个张量，而keep_prod是一个（0,1）之间的值，x中的元素清零的概率相互独立，为1-keep_prod,其他的元素按照1/keep_prod重新计算概率；
# a = tf.get_variable('a', shape=[2,5])
# b=a
# a_drop=tf.nn.dropout(a,0.8)
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# print(sess.run(b))
# print(sess.run(a_drop))


# mnist的应用：
# data_sets = input_data.read_data_sets('E:/Python_workspace/mnist-master/Mnist_data/')
# images = data_sets.train.images
# labels = data_sets.train.labels
# total = images.shape[0]
# print(images.shape)
# print(images)
# im = images[7]
# im2 = np.array(im)
# print(im)
# im2 = im2.reshape(28,28)
# print(im2)
# fig = plt.figure()
# plotwindow = fig.add_subplot(1,1,1)
# plt.imshow(im2, cmap='gray')
# plt.show()

# 计算某一张量的取值；类似于x.value()
# a = tf.constant([1.0, 2.0], name="a")
# b = tf.constant([2.0, 3.0], name="b")
# c = tf.add(a, b, name="sum")
# sess = tf.Session()
# with sess.as_default():
#     print(c.eval())

















