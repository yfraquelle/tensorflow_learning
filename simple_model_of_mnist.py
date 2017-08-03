import tensorflow as tf
import input_data

## 加载数据, mnist作为一个类，以numpy数组的形式存储训练、校验和测试的数据集，并提供一个函数用于在迭代中获得minibatch
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
## 数据下载完毕

##设置模型
# x是784维图像的输入占位，None表示任意长度
x = tf.placeholder("float", [None, 784])
# 权重： 784维图像 * W = 10维证据, 每一位表示0~9一个数字的权重 
W = tf.Variable(tf.zeros([784,10]))
# 偏移量： 10维向量，每一位表示0~9一个数字的偏移量
b = tf.Variable(tf.zeros([10]))
# x * W + b 作为softmax函数的输入，得到 0~9 数字的预测概率分布
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 真实的分布的输入占位
y_ = tf.placeholder("float",[None, 10])
# 交叉熵： 衡量预测分布的低效性
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 用梯度下降算法以0.01的学习速率最小化交叉熵（还有其他算法）
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
## 模型设置完毕

## 训练模型
# 初始化创建的变量
init = tf.initialize_all_variables()
# 在一个session中启动模型：tensorflow使用C++后端来计算，session是与后端的连接
sess = tf.Session()
sess.run(init)
# 训练模型，循环1000次————随机梯度下降训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #随机取训练数据中100个数据点
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys}) #将100个数据点作为参数替换之前的占位符来运行train_step
## 模型训练完毕

## 评估模型
# tf.argmax得到某个tensor对象在某一维上的数据最大值所在的索引，即类别标签； tf.equal检测预测与真实是否匹配;得到一组布尔值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 转化为0和1，求出平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
# 计算学习的模型在测试数据集上的正确率
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
