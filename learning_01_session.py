# tutorial code here https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT

import tensorflow as tf

matrix1 = tf.constant([3,3])
matrix2 = tf.constant([2],
                      [2])
# 矩阵乘法
product = tf.matmul(matrix1,matrix2)

#session执行命令————对话控制
#方法一
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

#方法二
with tf.Session() as sess:#打开session用sess来命名，最后自动关闭
    result2=sess.run(product)
    print(result2)
