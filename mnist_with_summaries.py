#!/usr/bin/env python
# _*_ UTF-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform
import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')


def main(_):
  # Import data
  mnist = input_data.read_data_sets('Mnist_data/', one_hot=True,fake_data=FLAGS.fake_data)
  # 能够在运行图的时候插入一些计算图
  sess = tf.InteractiveSession()

  # Create the model
  # placeholder：定义形参
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  W = tf.Variable(tf.zeros([784, 10]), name='weights')
  b = tf.Variable(tf.zeros([10], name='bias'))

  # Use a name scope to organize nodes in the graph visualizer
  with tf.name_scope('Wx_b'):
    y = tf.nn.softmax(tf.matmul(x, W) + b)

  # Add summary ops to collect data
  # 将数据分布以直方图的形式呈现在仪表盘上
  _ = tf.summary.histogram('weights', W)
  _ = tf.summary.histogram('biases', b)
  _ = tf.summary.histogram('y', y)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
  # More name scopes will clean up the graph representation
  with tf.name_scope('xent'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#     tf.summary.scalar：用来显示标量信息
    _ = tf.summary.scalar('cross entropy', cross_entropy)
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(
        FLAGS.learning_rate).minimize(cross_entropy)

  with tf.name_scope('test'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _ = tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter('/tmp/mnist_logs', sess.graph_def)
  tf.initialize_all_variables().run()

  # Train the model, and feed in test data and record summaries every 10 steps

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summary data and the accuracy
      if FLAGS.fake_data:
        batch_xs, batch_ys = mnist.train.next_batch(
            100, fake_data=FLAGS.fake_data)
        feed = {x: batch_xs, y_: batch_ys}
      else:
        feed = {x: mnist.test.images, y_: mnist.test.labels}
      result = sess.run([merged, accuracy], feed_dict=feed)
      summary_str = result[0]
      acc = result[1]
      writer.add_summary(summary_str, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      batch_xs, batch_ys = mnist.train.next_batch(
          100, fake_data=FLAGS.fake_data)
      feed = {x: batch_xs, y_: batch_ys}
      sess.run(train_step, feed_dict=feed)

if __name__ == '__main__':
  tf.app.run()
