import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## 设置模型
# 输入图像和预测概率分布的占位
x = tf.placeholder("float", shape=[None,784])
y = tf.placeholder("float", shape=[None,10])
y_ = tf.placeholder("float",[None, 10])
# 权重和偏置，初始化为0
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
## 模型设置完毕

## 训练模型
# 若在python外部计算，减少切换开销
sess = tf.InteractiveSession()
# 通过session初始化
sess.run(tf.initialize_all_variables())
# 类别预测
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 损失函数，每张图片的交叉熵值之和
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 梯度下降法，步长0.01，优化参数，返回的train_step对象在运行时梯度下降来更新参数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 训练
for i in range(1000):
    batch = mnist.train.next_batch(50) # 每次迭代加载50个训练样本，执行一次train_step，通过feed_dict将x和y_占位符用训练数据替代
    train_step.run(feed_dict = {x:batch[0],y_:batch[1]})    
## 模型训练完毕

## 评估模型
# 预测标签与真实标签的符合情况数组
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# 转化为0和1，并用平均值表示准确性
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
## 模型评估完毕




