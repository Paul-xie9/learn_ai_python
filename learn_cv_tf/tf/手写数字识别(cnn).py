# cnn卷积神经运算

import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 定义输入
imageInput = tf.placeholder(tf.float32, [None, 784])
labeInput = tf.placeholder(tf.float32, [None, 10])
# 数据转换形状    [None,784]->M*28*28*1   2D->4D  28*28:表示宽和高 1:表示通道
imageInputReshape = tf.reshape(imageInput, [-1, 28, 28, 1])
# 卷积
# 权重w0,卷积核:5*5,输出:32,输入:1;  方差:0.1
w0 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
# 偏移矩阵
b0 = tf.Variable(tf.constant(0.1, shape=[32]))
# 卷积运算
# 卷积层
layer1 = tf.nn.relu(tf.nn.conv2d(imageInputReshape, w0, strides=[1, 1, 1, 1], padding='MASE'))
# 池化层;采样作用,下采样,数据量不断减小  M*28*28*32=>M*7*7*32(输出结果)
layer1_pool = tf.nn.max_pool(layer1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='MASE')
