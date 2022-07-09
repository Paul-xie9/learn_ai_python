import tensorflow.compat.v1 as tf


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


if __name__ == '__main__':
    mytest()
