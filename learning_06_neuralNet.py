import tensorflow as tf
import numpy as np
# 添加神经层的函数
def add_layer(inputs,in_size,out_size,activation_function=None):# None means linear
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))# in_size行，out_size列的矩阵
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)# 向量，初始值最好不为0
    Wx_plus_b=tf.matmul(inputs,Weights)+biases#矩阵乘法加上偏移就是预测值
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
    
x_data=np.linspace(-1,1,300)[:,np.newaxis] # -1~1之间300个单位，并指定维度
noise=np.random.normal(0,0.05,x_data.shape) # 噪点，均值0,方差0.05，与x_data一样的格式
y_data=np.square(x_data)-0.5+noise

# 给train_step的值，数据只有1个，可以给任意个例子
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
# 输入层输入多少个data就有多少神经元、隐藏层假设10个神经元、，输出层输出多少个data就有多少神经元
# 隐藏层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu) # 输入数据为x_data、输入一个数据、输出10个数据（隐藏层有10个神经元）
# 输出层
prediction=add_layer(l1, 10, 1, activation_function=None) # 线性函数

# 预测之前，计算预测值与真实值的差别
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1])) # 实际算出的是每个例子的平方，需要求和，再求出平均值

# 训练，提升误差，选择了一个最常用的Optimizer，需要指定学习效率，一般小于1，Optimizer用于减小误差
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化所有变量
init=tf.initialize_all_variables()
# 定义session
sess=tf.Session()
# 开始运算
sess.run(init)
# 学习重复1000次
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data}) # 这样传入可以只传入一部分数据，小步骤训练可以提升效率=，这里假设用全部的数据来提升运算
    if i % 50==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
    