import tensorflow as tf
import input_data
from learning_06_neuralNet import prediction
from convolutional_net_model_of_mnist import cross_entropy, correct_prediction

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 之前解决的都是线性回归的问题，值可以是连续的
# mnist中x为784向量（图片），y为10向量（数字）

def add_layer(inputs,in_size,out_size,activation_function=None):# None means linear
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))# in_size行，out_size列的矩阵
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)# 向量，初始值最好不为0
    Wx_plus_b=tf.matmul(inputs,Weights)+biases#矩阵乘法加上偏移就是预测值
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction # 将prediction定义为全局变量
    y_pre=sess.run(prediction,feed_dict={xs:v_ys}) # y的预测值，是一个0~1的预测概率
    correct_prediction=tf.equal(tr.argmax(y_pre,1),tf.arg_max(v_ys,1)) # 对比预测值和真实值的差别，当预测概率最大时是否为真实值的概率为1时的值
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 整个数据库中多少是对的多少是错的
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys}) # 输出一个百分比，越高准确率越高
    return result
    
# 定义placeholder
xs=tf.placeholder(tf.float32,[None,784]) # 28x28
ys=tf.placeholder(tf.float32,[None,10])

# 添加输出层
prediction=add_layer(xs,784,10, activation_function=tf.nn.softmax) # 常用于分类

# 预测和真实之间的差值->loss(cross_entropy算法)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)),reduction_indices=[1])
# 训练
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Session
sess=tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100) # 每次提取100个数据，小步骤学习
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels)) # 计算准确度
        
        