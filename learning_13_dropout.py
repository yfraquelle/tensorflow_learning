import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# python需要的lib大部分可以在这里找到
# http://www.lfd.uci.edu/~gohlke/pythonlibs
# 最好使用power shell来安装
# pip install sklearn
# pip install scipy (可能失败，下载whl文件，进入保存的文件夹再pip install scipy-0.19.0...whl)
# 如果显示下述提示，可能是numpy已经安装，但是scipy用预编译的版本安装，需要numpy+mkl
# from numpy._distributor_init import NUMPY_MKL  # requires numpy+mkl
# ImportError: cannot import name 'NUMPY_MKL'
# pip install num....whl

# 过拟合，希望误差最小，但不能成功表达训练数据之外的数据
# 方法三：Dropout正规化，每次训练后随机忽略一些神经元，使得神经网络不完整，这样得出的结果不会依赖于某一部分特定的神经元

# 加载数据
digits=load_digits()
X=digits.data # 从0~9的图片数字
y=digits.target # 长度为10的向量
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

def add_layer(inputs,in_size,out_size,name_layer,activation_function=None):# None means linear
    layer_name='layer%s'%name_layer
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))# in_size行，out_size列的矩阵
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)# 向量，初始值最好不为0
    Wx_plus_b=tf.matmul(inputs,Weights)+biases#矩阵乘法加上偏移就是预测值
    # 需要将Wx_plus_b的一部分dropout
    Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)######！
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    # 需要有一个histogram summary来表示outputs
    tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs
    
keep_prob=tf.placeholder(tf.float32) # 保持不被dropout的百分比######！
xs=tf.placeholder(tf.float32,[None,64]) # 8x8
ys=tf.placeholder(tf.float32,[None,10])

# 添加输入层
l1=add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax) # 隐藏层输入过多可能报错，这里使用50
# 预测和真实之间的差值->loss(cross_entropy算法)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

# 需要有一个scalar summary来表示loss
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess=tf.Session()
# 所有的summary
merge=tf.summary.merge_all()

# 文件存在log/train和logs/test文件夹中
train_writer=tf.summary.FileWriter("logs/train",sess.graph)
test_writer=tf.summary.FileWriter("logs/test",sess.graph)

sess.run(tf.initialize_all_variables())

for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.6}) # 保持60%的概率不被drop,即drop掉40%######！
    if i%50==0:
        # 每50步记录训练和测试结果存入所有的summary
        train_result=sess.run(merge,feed_dict={xs:X_train,ys:y_train,keep_prob:1}) # test的过程中不drop任何######！
        test_result=sess.run(merge,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        # 加载到writer中
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)
        
        
        