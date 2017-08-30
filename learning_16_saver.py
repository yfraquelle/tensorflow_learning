import tensorflow as tf

# saver tensorflow现在只能保存Variable不能保存整个神经网络框架
W=tf.Variable([[1,2,3],[4,5,6]],dtype=tf.float32,name='weights')
b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases') # dtype需要一样
 
init=tf.initialize_all_variables()
saver=tf.train.Saver()
 
with tf.Session() as sess:
    sess.run(init)
    save_path=saver.save(sess,"net/save.ckpt") # 将整个sess中的内容保存到与py同根目录下的net文件夹(先创建好)中的save.ckpt


# restore
import numpy as np

# 重新定义与神经网络一样的shape和type
W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights") # np定义6个值得向量，reshape成2x3
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")

# 不需要定义init
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"net/save.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))