import random

import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd


# 最初版
def sampleCNN_1():
    # 1.输入层节点
    x1 = tf.placeholder(dtype=tf.float32)
    x2 = tf.placeholder(dtype=tf.float32)
    x3 = tf.placeholder(dtype=tf.float32)
    # 目标计算结果
    yTrain = tf.placeholder(dtype=tf.float32)

    # 神经网络的可变参数
    w1 = tf.Variable(0.1, dtype=tf.float32)
    w2 = tf.Variable(0.1, dtype=tf.float32)
    w3 = tf.Variable(0.1, dtype=tf.float32)

    # 3.隐藏层的计算
    n1 = x1 * w1
    n2 = x2 * w2
    n3 = x3 * w3

    # 4.输出层
    y = n1 + n2 + n3

    # 训练这个神经网络
    # 计算神经网络的计算结果y与目标值yTrain之间的误差
    loss = tf.abs(y - yTrain)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(2000):
        result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss],
                          feed_dict={x1: 90, x2: 80, x3: 70, yTrain: 85})
        print('result1:', result)
    result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss], feed_dict={x1: 98, x2: 95, x3: 87, yTrain: 96})
    print('result2:', result)


# 将输入简化
def sampleCNN_2():
    # 输入参数
    x = tf.placeholder(shape=[3], dtype=tf.float32)
    yTrain = tf.placeholder(shape=[], dtype=tf.float32)
    w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
    # 隐藏层
    n = x * w
    # 作用是把作为它的参数的向量（以后还可能会是矩阵）中的所有维度的值相加求和
    y = tf.reduce_sum(n)
    loss = tf.abs(y - yTrain)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(5000):
        result = sess.run([train, x, w, y, yTrain, loss], feed_dict={x: [90, 80, 70], yTrain: 85})
        print("x: [90, 80, 70], yTrain: 85=>", result)
        result = sess.run([train, x, w, y, yTrain, loss], feed_dict={x: [98, 95, 87], yTrain: 96})
        print("x: [98, 95, 87], yTrain: 96=>", result)


# 加上softmax函数
def sampleCNN_3():
    # 输入参数
    x = tf.placeholder(shape=[3], dtype=tf.float32)
    yTrain = tf.placeholder(shape=[], dtype=tf.float32)
    w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
    # 它可以把一个向量规范化后得到一个新的向量，这个新的向量中的所有数值相加起来保证为1
    wn = tf.nn.softmax(w)
    # 隐藏层
    n = x * wn
    # 作用是把作为它的参数的向量（以后还可能会是矩阵）中的所有维度的值相加求和
    y = tf.reduce_sum(n)
    loss = tf.abs(y - yTrain)
    # 学习效率
    optimizer = tf.train.RMSPropOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(5):
        result = sess.run([train, x, w, wn, y, yTrain, loss], feed_dict={x: [90, 80, 70], yTrain: 85})
        # print("x: [90, 80, 70], yTrain: 85=>", result)
        print("x: [90, 80, 70], yTrain: 85=>", result[3])
        result = sess.run([train, x, w, wn, y, yTrain, loss], feed_dict={x: [98, 95, 87], yTrain: 96})
        # print("x: [98, 95, 87], yTrain: 96=>", result)
        print("x: [98, 95, 87], yTrain: 96=>", result[3])
    sess.close()


# 测试张量
def mytest():
    x = tf.placeholder(shape=[3, 2], dtype=tf.float32)
    m = x * 7
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run([x, m], feed_dict={x: [[3, 2], [5, 6], [4, 7]]})
    print("[3,2]的张量与7相乘的结果=>", result[1])


# sigmoid激活函数将线性问题处理为非线性问题
def sigmoidTest():
    # 输入层
    x = tf.placeholder(dtype=tf.float32)
    # 目标值
    yTrain = tf.placeholder(dtype=tf.float32)
    # 隐藏层
    w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
    # 加入偏移量加快训练速度
    b = tf.Variable(80, dtype=tf.float32)
    wn = tf.nn.softmax(w)
    n1 = wn * x
    n2 = tf.reduce_sum(n1) - b
    # sigmoid激活函数
    y = tf.nn.sigmoid(n2)
    loss = tf.abs(yTrain - y)
    optimizer = tf.train.RMSPropOptimizer(0.1)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    random.seed()  # 产生新的随机数种子
    # 伪造数据
    for i in range(5):
        # 成绩范围 [60,100]
        xData = [int(random.random() * 41 + 60), int(random.random() * 41 + 60), int(random.random() * 41 + 60)]
        xAll = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
        if xAll >= 95:
            yTrainData = 1
        else:
            yTrainData = 0
        result = sess.run([train, x, yTrain, w, b, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
        print("[train, x, yTrain, w, b, n2, y, loss]=>[60,100]=> ", result)
        # 优秀成绩范围 [93,100]
        xData = [int(random.random() * 8 + 93), int(random.random() * 8 + 93), int(random.random() * 8 + 93)]
        xAll = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
        if xAll >= 95:
            yTrainData = 1
        else:
            yTrainData = 0
        result = sess.run([train, x, yTrain, w, b, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
        print("[train, x, yTrain, w, b, n2, y, loss]=>[93,100]=> ", result)
        print()


# 批量生产随机数
def createMultiple():
    random.seed()
    rowCount = 5
    # np.full函数的作用是生成一个多维数组，并用预定的值来填充
    xData = np.full(shape=(rowCount, 3), fill_value=0, dtype=np.float32)
    yTrainData = np.full(shape=rowCount, fill_value=0, dtype=np.float32)
    goodCount = 0
    # 生产随机训练的数据
    for i in range(rowCount):
        xData[i][0] = int(random.random() * 11 + 90)
        xData[i][1] = int(random.random() * 11 + 90)
        xData[i][2] = int(random.random() * 11 + 90)
        xAll = xData[i][0] * 0.6 + xData[i][1] * 0.3 + xData[i][2] * 0.1
        if xAll >= 95:
            yTrainData[i] = 1
            # goodCount用来记录符合三好学生条件的数据的个数
            goodCount += 1
        else:
            yTrainData[i] = 0
    print("xData=> ", xData)
    print("yTrainData=> ", yTrainData)
    print("goodCount=> ", goodCount)
    x = tf.placeholder(dtype=tf.float32)
    yTrain = tf.placeholder(dtype=tf.float32)
    w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
    b = tf.Variable(80, dtype=tf.float32)
    wn = tf.nn.softmax(w)
    n1 = wn * x
    n2 = tf.reduce_sum(n1) - b
    y = tf.nn.sigmoid(n2)
    loss = tf.abs(yTrain - y)
    optimizer = tf.train.RMSPropOptimizer(0.1)
    train = optimizer.minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        for j in range(rowCount):
            result = sess.run([train, x, yTrain, wn, b, n2, y, loss], feed_dict={x: xData[j], yTrain: yTrainData[j]})
            print("[train,x,yTrain,wn,b,n2,y,loss]=> ", result)


# 从外部文件中批量读取训练数据
def extensionFile():
    filePath = "./file/data.txt"
    # np.loadtxt()第一个参数代表要读取的文件名，命名参数delimiter表示数据项之间用什么字符分隔，命名参数dtype表示读取的数据类型
    wholeData = np.loadtxt(filePath, delimiter=",", dtype=np.float32)
    print(wholeData)


# 利用pandas处理带中文的字符串文件
def pandasLoadFile(fileName):
    fileName = './file/data.csv'
    # 该函数中的命名参数header要传入None（这是Python中用来代表“没有”的一个特殊的值）
    # 否则会把文件的第一行当表头文字来处理；命名参数usecols是一个用圆括号括起来的数字集合，代表希望读取每一行中的哪一些列，
    fileData = pd.read_csv(fileName, dtype=np.float32, header=None, usecols=(1, 2, 3, 4))
    wholeData = fileData.values
    print(wholeData)
    return wholeData


# 从外部引入数据训练
def extensionDataCNN():
    data = pandasLoadFile(None)
    rowCount = int(data.size / data[0].size)
    goodCount = 0
    for i in range(rowCount):
        if data[i][0] * 0.6 + data[i][1] * 0.3 + data[i][2] * 0.1 >= 95:
            goodCount += 1
    print("data=>", data)
    print("rowCount=>", rowCount)
    print("goodCount=>", goodCount)
    # 输入层
    x = tf.placeholder(dtype=tf.float32)
    # 目标值
    yTrain = tf.placeholder(dtype=tf.float32)
    # 隐藏层
    w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
    # 加入偏移量加快训练速度
    b = tf.Variable(80, dtype=tf.float32)
    wn = tf.nn.softmax(w)
    n1 = wn * x
    n2 = tf.reduce_sum(n1) - b
    # sigmoid激活函数
    y = tf.nn.sigmoid(n2)
    loss = tf.abs(yTrain - y)
    optimizer = tf.train.RMSPropOptimizer(0.1)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(2):
        for j in range(rowCount):
            result = sess.run([train, x, yTrain, wn, b, n2, y, loss], feed_dict={x: data[j][0:3], yTrain: data[j][3]})
            print("result", result)


# 判断身份证性别（单层神经网络）
def identityGender():
    x = tf.placeholder(tf.float32)
    yTrain = tf.placeholder(tf.float32)
    # 可变参数w
    # random_normal函数产生的随机数是符合数学中正态分布的概率的,mean是指定这个平均值的，stddev是指定这个波动范围的
    w = tf.Variable(tf.random_normal([4], mean=0.5, stddev=0.1), dtype=tf.float32)
    # 偏移量b
    b = tf.Variable(0, dtype=tf.float32)
    n1 = w * x + b
    y = tf.nn.sigmoid(tf.reduce_sum(n1))
    loss = tf.abs(y - yTrain)
    optimizer = tf.train.RMSPropOptimizer(0.01)
    train = optimizer.minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # lossSum变量来记录训练中误差的总和
    lossSum = 0.0
    for i in range(100000):
        xDataRandom = [int(random.random() * 10), int(random.random() * 10), int(random.random() * 10),
                       int(random.random() * 10)]
        if xDataRandom[2] % 2 == 0:
            yTrainDataRandom = 0
        else:
            yTrainDataRandom = 1
        result = sess.run([train, x, yTrain, y, loss], feed_dict={x: xDataRandom, yTrain: yTrainDataRandom})
        lossSum = lossSum + float(result[len(result) - 1])
        print("i: %d, loss: %10.10f, avgLoss: %10.10f" % (i, float(result[len(result) - 1]), lossSum / (i + 1)))


if __name__ == '__main__':
    identityGender()
