import tensorflow as tf

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
    result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss], feed_dict={x1: 90, x2: 80, x3: 70, yTrain: 85})
    print('result1:', result)
result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss], feed_dict={x1: 98, x2: 95, x3: 87, yTrain: 96})
print('result2:', result)
