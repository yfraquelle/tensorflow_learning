import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32) # 输入值
y_data = x_data*0.1 + 0.3 # 正确的函数

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 权重，起始值为-1~1之间的一个随机值，通过训练希望接近0.1
biases = tf.Variable(tf.zeros([1])) # 偏移量，起始值为0，通过训练希望接近0.3

y = Weights*x_data + biases # 模型的初始形式

loss = tf.reduce_mean(tf.square(y-y_data)) # 成本函数，表示预测值与正确值之间的差值，
optimizer = tf.train.GradientDescentOptimizer(0.5) # 梯度下降法最小化差值，学习效率为0.5
train = optimizer.minimize(loss)

init = tf.initialize_all_variables() # 初始化所有变量

sess = tf.Session() # 定义session
sess.run(init) # 执行初始化

for step in range(201):
    sess.run(train) # 执行训练过程
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))