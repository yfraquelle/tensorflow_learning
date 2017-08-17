import tensorflow as tf

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
    