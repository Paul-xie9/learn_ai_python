import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# knn：最近领域法，将样本和手写数字进行比较，找出k个相似的样本，再在k个样本中选出概率最大的一个
# 1.装载数据
mnist = input_data.read_data_sets('../resource/MNTST_data', one_hot=True)
# 2.属性设置
trainNum = 55000  # 训练数
testNum = 10000  # 测试数
trainSize = 500
testSize = 5
k = 4
# 3.数据分解
trainIndex = np.random.choice(trainNum, trainSize, replace=False)
testIndex = np.random.choice(testNum, testSize, replace=False)
trainData = mnist.train.images[trainIndex]  # 训练图片
trainLabel = mnist.train.labels[trainIndex]  # 训练标签
testData = mnist.test.images[testIndex]  # 测试图片
testLabel = mnist.test.labels[testIndex]  # 测试标签
# 28*28 = 784
print('trainData.shape=', trainData.shape)  # 500*784 1 图片个数 2 784?
print('trainLabel.shape=', trainLabel.shape)  # 500*10
print('testData.shape=', testData.shape)  # 5*784
print('testLabel.shape=', testLabel.shape)  # 5*10
print('testLabel=', testLabel)  # 4 :testData [0]  3:testData[1] 6
# 4.输入数据规范
trainDataInput = tf.placeholder(shape=[None, trainData.shape[1]], dtype=tf.float32)
trainLabelInput = tf.placeholder(shape=[None, testLabel.shape[1]], dtype=tf.float32)
testDataInput = tf.placeholder(shape=[None, testData.shape[1]], dtype=tf.float32)
testLabelInput = tf.placeholder(shape=[None, testLabel.shape[1]], dtype=tf.float32)
# 5.计算knn的距离
f1 = tf.expand_dims(testDataInput, 1)  # 维度扩展
f2 = tf.subtract(trainDataInput, f1)
f3 = tf.reduce_sum(tf.abs(f2), reduction_indices=2)  # 完成数据累加
f4 = tf.negative(f3)  # 取反
f5, f6 = tf.nn.top_k(f4, k=4)  # f5,f6是f3中最小的数值
f7 = tf.gather(trainLabelInput, f6)
f8 = tf.reduce_sum(f7, reduction_indices=1)
f9 = tf.argmax(f8, dimension=1)
with tf.Session() as sess:
    p1 = sess.run(f1, feed_dict={testDataInput: testData[0:5]})
    print('p1.shape: ', p1.shape)
    p2 = sess.run(f2, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p2.shape: ', p2.shape)
    p3 = sess.run(f3, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p3.shape: ', p3.shape)
    print('p3[0,0]: ', p3[0, 0])
    p4 = sess.run(f4, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p4.shape: ', p4.shape)
    print('p4[0,0]: ', p4[0, 0])
    p5, p6 = sess.run((f5, f6), feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p5.shape: ', p5.shape)
    print('p6.shape: ', p6.shape)
    print('p5[0,0]: ', p5[0])
    print('p6[0,0]: ', p6[0])
    p7 = sess.run(f7, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p7.shape: ', p7.shape)
    print('p7[]: ', p7)
    p8 = sess.run(f8, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p8.shape: ', p8.shape)
    print('p8[]: ', p8)
    p9 = sess.run(f9, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p9.shape: ', p9.shape)
    print('p9[]: ', p9)
    p10 = np.argmax(testLabel[0:5], axis=1)
    print('p10[]: ', p10)
# 检测概率统计
j = 0
for i in range(0, 5):
    if p10[i] == p9[i]:
        j = j + 1
print('ac: ', j * 100 / 5)
