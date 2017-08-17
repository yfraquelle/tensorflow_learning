import tensorflow as tf

state=tf.Variable(0,name='counter')
#print(state.name)
one=tf.constant(1)
#操作
new_value=tf.add(state,one)
update=tf.assign(state,new_value)
#定义变量后必须初始化
init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))