import tensorflow as tf
from sklearn.datasets import load_digits
# 过拟合，希望误差最小，但不能成功表达训练数据之外的数据
# 方法三：Dropout正规化，每次训练后随机忽略一些神经元，使得神经网络不完整，这样得出的结果不会依赖于某一部分特定的神经元
