#!/usr/bin/env python
# _*_ UTF-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

# 一：用numpy写的一个神经网络流程：
# # 数据集为：
# # 001  0
# # 111  1
# # 101  1
# # 011  0
# # 前三个维度为x，第四个维度为y；
# # 输入神经元有三个维度，得到一个y值：f(w1*x1+w2*x2+w3*x3)=y
# # 激活函数为：sigmod函数；
# # 主要的计算过程为：1）前向传播求损失；2）后向传播求权值；
# # 一个主要的求导公式为：dy/dx=y(1-y)
# # 构建逻辑回归函数：
# def nonlin(x, deriv=False):
#     if(deriv == True):
#         return x*(1-x)
#     return 1/(1+np.exp(-x))
# # 输入值：
# x = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
# # 输出值：
# y = np.array([0,1,1,1]).T
# # 初始化权重syn0：
# # seed(num)是产生随机数：如果num的数据一致，则产生的随机数一样；如果不一致，则产生的随机数不一样；如果括号中为空，则seed()随机产生随机数；
# np.random.seed(1)
# syn0 = 2*np.random.random((3,1))-1
# # 迭代系统：
# for iter in range(10000):
#     l0 = x
#     # l1:是运算结果；
#     l1 = nonlin(np.dot(l0, syn0))
#     y1 = [[0 for i in range(1)] for i in range(len(y))]
#     for i in range(len(y)):
#         y1[i][0] = y[i]
#     # 运算误差
#     l1_error = y1 - l1
#     # l1_delta:权重变化值；
#     # nonlin(l1, True):每一组的导数值；
#     l1_delta =l1_error*nonlin(l1, True)
#     syn0 += np.dot(l0.T, l1_delta)
# #     print(syn0)
# print("output after training")
# print(l1)
# # l1:是经过四组数据训练后的最终结果；四次训练后会得到一个稳定的权重值，然后用这组权重值syn0进行分类；
# 二、一个简单的tensorflow样式：
# # 构造数据：
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data * 0.1 + 0.3
# # 定义变量：
# # 生成随机的权重：[1]：两句话中的内容为向量的形状，即一维向量；
# weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# biases = tf.Variable(tf.zeros([1]))
# y = weight*x_data + biases
# # 计算损失：
# # reduce_mean：计算所有值和的均值
# loss = tf.reduce_mean(tf.square(y-y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# # 每次迭代0.5，然后计算loss，当出现最小loss时停止；
# train = optimizer.minimize(loss)
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# for step in range(400):
#     sess.run(train)
#     if step%20 == 0:
#         print(step, sess.run(weight), sess.run(biases))
# 三、tensorflow应用：

# 主要分为两个步骤：1）构建模型；2）训练；
# 1)构建一个tensor：
# a = tf.zeros([2,3])
# print(a)
# sess = tf.InteractiveSession()
# print(sess.run(a))
# 2）图中的计算参数variable,需要进行初始化才能使用；
# 构建模型：
# w=tf.Variable(tf.zeros([3,3]))
# # 构建session：
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# print(sess.run(w))
# 3）占位符：placeholder： 
# 一般在输入时使用，表示有值输入：
# 4）session：
# 抽象模型的实现者，模型构建好之后，需要构建一个session才能够运行模型；
# 5）模型构建：
# 定义模型：
# 定义损失函数和训练方法：
# reduction_indices=[1]：横向求和；
# reduction_indices=[0]：纵向求和；
# from tensorflow.examples.tutorials.mnist import input_data
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('data_dir', '/Mnist_data/', 'Directory for storing data')
# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
# # 需要多少层、每层有多少个节点，多个案例循环处理，得到多组分类，然后多个结果使用混淆矩阵，判断哪个的效果比较好；
# # 模型构建：
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
# w = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
# a = tf.nn.softmax(tf.matmul(x, w)+b)
# # 模型调参：
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(a), reduction_indices=[1]))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(cross_entropy)
# # 测试模型：
# correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# # 开始训练：
# sess = tf.InteractiveSession()
# # 初始化所有变量：
# tf.initialize_all_variables().run()
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     train.run({x:batch_xs, y:batch_ys})
# print(sess.run(accuracy, feed_dict={x:mnist.test.image, y:mnist.test.labels}))
# 手写识别：
# sess = tf.InteractiveSession()
# tf.initialize_all_variables().run()
# for i in range(100):
#     batch_xs, batch_ys = mnist.train.next_batch(10)
#     train.run({x:batch_xs, y:batch_ys})
# 
# prediction=tf.argmax(a,1)
# print(prediction)
# predint=prediction.eval(feed_dict={x:[result],keep_prob:1.0}, session=sess)
# 
# print(prediction.eval())
# print(predint)

# 参数的循环组合；
# 输入的数据；
# 训练的数据；
# 建立的模型；
# 开始训练；
# 开始预测；
# 混淆矩阵；
# 四、CNN案例：
# tf.nn.conv2d:给定四维的input和filter，计算出两维的结果；
# tf.nn.max_pool:最大值池化操作；
# Import data
# from tensorflow.examples.tutorials.mnist import input_data
# 
# import tensorflow as tf
# 
# mnist = input_data.read_data_sets('E:/Python_workspace/mnist-master/Mnist_data/', one_hot=True)
# 
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
#     return tf.Variable(initial)
# 
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
# 
# def conv2d(x, W):
#     """
#     tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
#     前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
#     input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
#     filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
#     strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
#     padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
#     use_cudnn_on_gpu 是否使用cudnn加速。默认是True
#     """
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 
# def max_pool_2x2(x):
#     """
#     tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
#     几个参数分别是：value, ksize, strides, padding,
#     value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
#     ksize:  长为4的list,表示池化窗口的尺寸
#     strides: 窗口的滑动值，与conv2d中的一样
#     padding: 与conv2d中用法一样。
#     """
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 
# sess = tf.InteractiveSession()
# x = tf.placeholder(tf.float32, [None, 784])
# x_image = tf.reshape(x, [-1,28,28,1]) #将输入按照 conv2d中input的格式来reshape，reshape
# """
# # 第一层
# # 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
# # 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*28*28*32
# # 也就是单个通道输出为28*28，共有32个通道,共有?个批次
# # 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*14*14*32
# """
# W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
# 
# """
# # 第二层
# # 卷积核5*5，输入通道为32，输出通道为64。
# # 卷积前图像的尺寸为 ?*14*14*32， 卷积后为?*14*14*64
# # 池化后，输出的图像尺寸为?*7*7*64
# """
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
# 
# # 第三层 是个全连接层,输入维数7*7*64, 输出维数为1024
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 
# # 第四层，输入1024维，输出10维，也就是具体的0~9分类
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 使用softmax作为多分类激活函数
# y_ = tf.placeholder(tf.float32, [None, 10])
# 
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # 损失函数，交叉熵
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 使用adam优化
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables()) # 变量初始化
# for i in range(20000):
#     batch = mnist.train.next_batch(50)
#     if i%100 == 0:
#         # print(batch[1].shape)
#         train_accuracy = accuracy.eval(feed_dict={
#             x:batch[0], y_: batch[1], keep_prob: 1.0})
#         print("step %d, training accuracy %g"%(i, train_accuracy))
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# 
# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# with tf.Session() as sess:
#     sess.run(init_op)
#     for i in range(100):
#         batch_xs, batch_ys = mnist.train.next_batch(100)
#         train.run({x:batch_xs, y:batch_ys})
#     prediction=tf.argmax(y_conv,1)
#     predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
#     print(h_conv2)
#     print('recognize result:')
#     print(predint[0])

# 五、LSTM代码：
# import time
# from tensorflow.models.rnn.ptb import reader
# 
# flags = tf.flags
# logging = tf.logging
# 
# flags.DEFINE_string(
#     "model", "small",
#     "A type of model. Possible options are: small, medium, large.")
# flags.DEFINE_string("data_path", '/home/multiangle/download/simple-examples/data/', "data_path")
# flags.DEFINE_bool("use_fp16", False,
#                   "Train using 16-bit floats instead of 32bit floats")
# 
# FLAGS = flags.FLAGS
# 
# 
# def data_type():
#     return tf.float16 if FLAGS.use_fp16 else tf.float32
# 
# 
# class PTBModel(object):
#     """The PTB model."""
# 
#     def __init__(self, is_training, config):
#         """
#         :param is_training: 是否要进行训练.如果is_training=False,则不会进行参数的修正。
#         """
#         self.batch_size = batch_size = config.batch_size
#         self.num_steps = num_steps = config.num_steps
#         size = config.hidden_size
#         vocab_size = config.vocab_size
# 
#         self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])    # 输入
#         self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])       # 预期输出，两者都是index序列，长度为num_step
# 
#         # Slightly better results can be obtained with forget gate biases
#         # initialized to 1 but the hyperparameters of the model would need to be
#         # different than reported in the paper.
#         lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
#         if is_training and config.keep_prob < 1: # 在外面包裹一层dropout
#             lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
#                 lstm_cell, output_keep_prob=config.keep_prob)
#         cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True) # 多层lstm cell 堆叠起来
# 
#         self._initial_state = cell.zero_state(batch_size, data_type()) # 参数初始化,rnn_cell.RNNCell.zero_state
# 
#         with tf.device("/cpu:0"):
#             embedding = tf.get_variable(
#                 "embedding", [vocab_size, size], dtype=data_type()) # vocab size * hidden size, 将单词转成embedding描述
#             # 将输入seq用embedding表示, shape=[batch, steps, hidden_size]
#             inputs = tf.nn.embedding_lookup(embedding, self._input_data)
# 
#         if is_training and config.keep_prob < 1:
#             inputs = tf.nn.dropout(inputs, config.keep_prob)
# 
#         # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
#         # This builds an unrolled LSTM for tutorial purposes only.
#         # In general, use the rnn() or state_saving_rnn() from rnn.py.
#         #
#         # The alternative version of the code below is:
#         #
#         # inputs = [tf.squeeze(input_, [1])
#         #           for input_ in tf.split(1, num_steps, inputs)]
#         # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
#         outputs = []
#         state = self._initial_state # state 表示 各个batch中的状态
#         with tf.variable_scope("RNN"):
#             for time_step in range(num_steps):
#                 if time_step > 0: tf.get_variable_scope().reuse_variables()
#                 # cell_out: [batch, hidden_size]
#                 (cell_output, state) = cell(inputs[:, time_step, :], state)
#                 outputs.append(cell_output)  # output: shape[num_steps][batch,hidden_size]
# 
#         # 把之前的list展开，成[batch, hidden_size*num_steps],然后 reshape, 成[batch*numsteps, hidden_size]
#         output = tf.reshape(tf.concat(1, outputs), [-1, size])
# 
#         # softmax_w , shape=[hidden_size, vocab_size], 用于将distributed表示的单词转化为one-hot表示
#         softmax_w = tf.get_variable(
#             "softmax_w", [size, vocab_size], dtype=data_type())
#         softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
#         # [batch*numsteps, vocab_size] 从隐藏语义转化成完全表示
#         logits = tf.matmul(output, softmax_w) + softmax_b
# 
#         # loss , shape=[batch*num_steps]
#         # 带权重的交叉熵计算
#         loss = tf.nn.seq2seq.sequence_loss_by_example(
#             [logits],   # output [batch*numsteps, vocab_size]
#             [tf.reshape(self._targets, [-1])],  # target, [batch_size, num_steps] 然后展开成一维【列表】
#             [tf.ones([batch_size * num_steps], dtype=data_type())]) # weight
#         self._cost = cost = tf.reduce_sum(loss) / batch_size # 计算得到平均每批batch的误差
#         self._final_state = state
# 
#         if not is_training:  # 如果没有训练，则不需要更新state的值。
#             return
# 
#         self._lr = tf.Variable(0.0, trainable=False)
#         tvars = tf.trainable_variables()
#         # clip_by_global_norm: 梯度衰减，具体算法为t_list[i] * clip_norm / max(global_norm, clip_norm)
#         # 这里gradients求导，ys和xs都是张量
#         # 返回一个长为len(xs)的张量，其中的每个元素都是\grad{\frac{dy}{dx}}
#         # clip_by_global_norm 用于控制梯度膨胀,前两个参数t_list, global_norm, 则
#         # t_list[i] * clip_norm / max(global_norm, clip_norm)
#         # 其中 global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
#         grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
#                                           config.max_grad_norm)
# 
#         # 梯度下降优化，指定学习速率
#         optimizer = tf.train.GradientDescentOptimizer(self._lr)
#         # optimizer = tf.train.AdamOptimizer()
#         # optimizer = tf.train.GradientDescentOptimizer(0.5)
#         self._train_op = optimizer.apply_gradients(zip(grads, tvars))  # 将梯度应用于变量
# 
#         self._new_lr = tf.placeholder(
#             tf.float32, shape=[], name="new_learning_rate")     #   用于外部向graph输入新的 lr值
#         self._lr_update = tf.assign(self._lr, self._new_lr)     #   使用new_lr来更新lr的值
# 
#     def assign_lr(self, session, lr_value):
#         # 使用 session 来调用 lr_update 操作
#         session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
# 
#     @property
#     def input_data(self):
#         return self._input_data
# 
#     @property
#     def targets(self):
#         return self._targets
# 
#     @property
#     def initial_state(self):
#         return self._initial_state
# 
#     @property
#     def cost(self):
#         return self._cost
# 
#     @property
#     def final_state(self):
#         return self._final_state
# 
#     @property
#     def lr(self):
#         return self._lr
# 
#     @property
#     def train_op(self):
#         return self._train_op
# 
# 
# class SmallConfig(object):
#     """Small config."""
#     init_scale = 0.1        #
#     learning_rate = 1.0     # 学习速率
#     max_grad_norm = 5       # 用于控制梯度膨胀，
#     num_layers = 2          # lstm层数
#     num_steps = 20          # 单个数据中，序列的长度。
#     hidden_size = 200       # 隐藏层规模
#     max_epoch = 4           # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
#     max_max_epoch = 13      # 指的是整个文本循环13遍。
#     keep_prob = 1.0
#     lr_decay = 0.5          # 学习速率衰减
#     batch_size = 20         # 每批数据的规模，每批有20个。
#     vocab_size = 10000      # 词典规模，总共10K个词
# 
# 
# class MediumConfig(object):
#     """Medium config."""
#     init_scale = 0.05
#     learning_rate = 1.0
#     max_grad_norm = 5
#     num_layers = 2
#     num_steps = 35
#     hidden_size = 650
#     max_epoch = 6
#     max_max_epoch = 39
#     keep_prob = 0.5
#     lr_decay = 0.8
#     batch_size = 20
#     vocab_size = 10000
# 
# 
# class LargeConfig(object):
#     """Large config."""
#     init_scale = 0.04
#     learning_rate = 1.0
#     max_grad_norm = 10
#     num_layers = 2
#     num_steps = 35
#     hidden_size = 1500
#     max_epoch = 14
#     max_max_epoch = 55
#     keep_prob = 0.35
#     lr_decay = 1 / 1.15
#     batch_size = 20
#     vocab_size = 10000
# 
# 
# class TestConfig(object):
#     """Tiny config, for testing."""
#     init_scale = 0.1
#     learning_rate = 1.0
#     max_grad_norm = 1
#     num_layers = 1
#     num_steps = 2
#     hidden_size = 2
#     max_epoch = 1
#     max_max_epoch = 1
#     keep_prob = 1.0
#     lr_decay = 0.5
#     batch_size = 20
#     vocab_size = 10000
# 
# 
# def run_epoch(session, model, data, eval_op, verbose=False):
#     """Runs the model on the given data."""
#     # epoch_size 表示批次总数。也就是说，需要向session喂这么多次数据
#     epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps  # // 表示整数除法
#     start_time = time.time()
#     costs = 0.0
#     iters = 0
#     state = session.run(model.initial_state)
#     for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
#                                                       model.num_steps)):
#         fetches = [model.cost, model.final_state, eval_op] # 要进行的操作，注意训练时和其他时候eval_op的区别
#         feed_dict = {}      # 设定input和target的值
#         feed_dict[model.input_data] = x
#         feed_dict[model.targets] = y
#         for i, (c, h) in enumerate(model.initial_state):
#             feed_dict[c] = state[i].c   # 这部分有什么用？看不懂
#             feed_dict[h] = state[i].h
#         cost, state, _ = session.run(fetches, feed_dict) # 运行session,获得cost和state
#         costs += cost   # 将 cost 累积
#         iters += model.num_steps
# 
#         if verbose and step % (epoch_size // 10) == 10:  # 也就是每个epoch要输出10个perplexity值
#             print("%.3f perplexity: %.3f speed: %.0f wps" %
#                   (step * 1.0 / epoch_size, np.exp(costs / iters),
#                    iters * model.batch_size / (time.time() - start_time)))
# 
#     return np.exp(costs / iters)
# 
# 
# def get_config():
#     if FLAGS.model == "small":
#         return SmallConfig()
#     elif FLAGS.model == "medium":
#         return MediumConfig()
#     elif FLAGS.model == "large":
#         return LargeConfig()
#     elif FLAGS.model == "test":
#         return TestConfig()
#     else:
#         raise ValueError("Invalid model: %s", FLAGS.model)
# 
# 
# # def main(_):
# if __name__=='__main__':
#     if not FLAGS.data_path:
#         raise ValueError("Must set --data_path to PTB data directory")
#     print(FLAGS.data_path)
# 
#     raw_data = reader.ptb_raw_data(FLAGS.data_path) # 获取原始数据
#     train_data, valid_data, test_data, _ = raw_data
# 
#     config = get_config()
#     eval_config = get_config()
#     eval_config.batch_size = 1
#     eval_config.num_steps = 1
# 
#     with tf.Graph().as_default(), tf.Session() as session:
#         initializer = tf.random_uniform_initializer(-config.init_scale, # 定义如何对参数变量初始化
#                                                     config.init_scale)
#         with tf.variable_scope("model", reuse=None,initializer=initializer):
#             m = PTBModel(is_training=True, config=config)   # 训练模型， is_trainable=True
#         with tf.variable_scope("model", reuse=True,initializer=initializer):
#             mvalid = PTBModel(is_training=False, config=config) #  交叉检验和测试模型，is_trainable=False
#             mtest = PTBModel(is_training=False, config=eval_config)
# 
#         summary_writer = tf.train.SummaryWriter('/tmp/lstm_logs',session.graph)
# 
#         tf.initialize_all_variables().run()  # 对参数变量初始化
# 
#         for i in range(config.max_max_epoch):   # 所有文本要重复多次进入模型训练
#             # learning rate 衰减
#             # 在 遍数小于max epoch时， lr_decay = 1 ; > max_epoch时， lr_decay = 0.5^(i-max_epoch)
#             lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
#             m.assign_lr(session, config.learning_rate * lr_decay) # 设置learning rate
# 
#             print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
#             train_perplexity = run_epoch(session, m, train_data, m.train_op,verbose=True) # 训练困惑度
#             print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
#             valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op()) # 检验困惑度
#             print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
# 
#         test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())  # 测试困惑度
#         print("Test Perplexity: %.3f" % test_perplexity)
# 
# 
# # if __name__ == "__main__":
# #     tf.app.run()
