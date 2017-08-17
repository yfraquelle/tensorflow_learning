import tensorflow as tf
# tf.train.GradientDescentOptimizer 最基础的线性优化，如果传入的数据只是一部分，就变成了SGD，一部分一部分地学习，可以更快到达全局最小量
# tf.train.AdadeltaOptimizer
# tf.train.AdagradOptimizer
# tf.train.MomentumOptimizer 常用，考虑了不仅本布的学习效率，还考虑了上一步的学习效率，可能一开始趋势错误，但能很快纠正，更快速到达全局最小量
# tf.train.AdamOptimizer 常用
# tf.train.FtrlOptimizer AlphaGO使用
# tf.train.RMSPropOptimizer


