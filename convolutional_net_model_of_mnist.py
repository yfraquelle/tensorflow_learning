import input_data
import tensorflow as tf

# 加入少量噪声来打破对称性和避免0梯度
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 使用ReLU神经元，用一个较小正数来初始化偏置，避免神经元节点输出恒为0的问题
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积，1步长、0边距
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
# 池化，2x2模板
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 加载数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## 设置模型
# 输入图像和预测概率分布的占位
x = tf.placeholder("float", shape=[None,784])
y = tf.placeholder("float", shape=[None,10])
y_ = tf.placeholder("float",[None, 10])
## 模型设置完毕

## 第一层：一个卷积接一个池化
# 卷积在每个5x5的patch中算出32个特征，权重向量为[5,5,1,32]，表示patch大小、输入通道数目、输出通道数目
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
# x变成一个4维向量，2、3维对应图片的宽、高，最后一维代表图片的颜色通道数
x_image = tf.reshape(x,[-1,28,28,1])
# 把x_image和权值向量进行卷积，加上偏置，用ReLU激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化
h_pool1 = max_pool_2x2(h_conv1)
## 第一层结束

## 第二层：将几个类似的层堆叠起来
# 每个5x5的patch得到64个特征
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
## 第二层结束

## 密集连接层：图片尺寸减小到7x7
# 加入一个有1024个神经元的全连接层，用于处理整个图片
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# 池化层输出tensor重置为一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
## 密集连接层结束

## Dropout：减少过拟合
# 一个神经元输出在dropout中保持不变的概率占位
keep_prob = tf.placeholder("float")
# 在训练过程中启用dropout，在测试过程中关闭dropout，tf.nn.dropout除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
## Dropout结束

## 输出层:softmax
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
## 输出层结束

## 训练模型
sess = tf.InteractiveSession()
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# ADAM优化器来做梯度最速下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
## 模型训练完毕

## 评估模型
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
## 模型训练完毕