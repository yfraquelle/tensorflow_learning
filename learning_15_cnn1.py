import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys): 
    global prediction 
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1}) 
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1}) 
    return result
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape) # 之后会从0.1变到其他值，一般用正值
    return tf.Variable(initial)
def conv2d(x,W): # 定义卷积神经网络层
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') # 输入为x参数，W为权重，stride为一个4元素向量，头尾必须是1，中间为水平和竖直方向的步长
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # max_pooling，从卷积层传出的传入pooling中，pooling的卷积核大小为2x2，步长相应为2，起到压缩图片的效果


# 定义神经网络的输入的placeholder
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1]) # xs包括了所有的图片例子，-1表示不管图片的维度，之后再定义,28,28表示像素点数，1表示一个通道（灰度图）
# print(x_image.shape) # 【n_samples,28,28,1】

# 定义第一个卷积层
W_conv1=weight_variable([5,5,1,32]) # 传入形状，输出Variable，5x5是patch，1是in_size（1个单位，灰度高度为1），32是out_size（高度为32）
b_conv1=bias_variable([32]) # 高度为32
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # 通过conv2d来处理x和W的关系，类似于y=W*x+b，并非线性化，输出大小为28x28x32(SAME不变长宽)
h_pool1=max_pool_2x2(h_conv1) # 输出大小为14x14x32（步长为2，长宽缩小1倍）
# 定义第二个卷积层
W_conv2=weight_variable([5,5,32,64]) # 传入形状，输出Variable，5x5是patch，32是in_size（32个单位），64是out_size（高度为64）
b_conv2=bias_variable([64]) # 高度为64
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) # 通过conv2d来处理x和W的关系，类似于y=W*x+b，并非线性化，输出大小为14x14x64(SAME不变长宽)
h_pool2=max_pool_2x2(h_conv2) # 输出大小为7x7x64（步长为2，长宽缩小1倍）

# 定义第一个神经网络层
W_fc1=weight_variable([7*7*64,1024]) # 输入的大小为第二层卷积层的输出形状(当作一维)，输出为1024让它变得更高
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64]) # {n_sample,7,7,64} >> [n_samples,7*7*64]
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1) #这一层的输出结果
h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob) # overfitting的处理
# 定义第二个神经网络层
W_fc2=weight_variable([1024,10]) # 输入的大小为第二层卷积层的输出形状(当作一维)，输出为1024让它变得更高
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2) #分类处理得到这一层的输出结果

# 预测值和真实值之间的差异
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess=tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50==0:
        print(compute_accuracy(mnist.test.images[:1000],mnist.test.labels[:1000]))
    