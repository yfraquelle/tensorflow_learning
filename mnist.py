# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.
    在此基础上借助ReLU激活函数，构建一对完全连接层，和一个有10个节点、指明了输出logtis模型的线性层
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1 每一层都创建一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀
  with tf.name_scope('hidden1'):
    # 在定义的作用域中，每一层所使用的权重和偏差都在tf.Variable实例中生成，并且包含了各自期望的shape
    # 每个变量在构建时都会获得初始化操作
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units], # 权重变量将图像输入链接到名为hidden1的第一层， tf.truncated_normal初始化权重变量，给赋予的shape是一个二维tensor，第一个维度表示该层中权重变量所连接的单元数量，第二个维度代表该层中权重变量所链接到的单元数量
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),#
        name='weights') # name will be hidden1/weights
    biases = tf.Variable(tf.zeros([hidden1_units]), # 通过tf.zeros初始化偏差变量，所有偏差的shape是其在该层中所接到的单元数量
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases) #1 图表的tf.nn.relu操作中嵌入了隐藏层所需的tf.matmul
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases) #2 
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases #3 logits模型所需的另外一个tf.matmul
  return logits # 返回包含了输出结果的logits Tensor
#1 #2 #3 依次生成，各自的tf.Variable实例与输入占位符或下一层的输出tensor所连接

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
    通过添加所需的损失操作进一步构建图表
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  # Convert from sparse integer labels in the range [0, NUM_CLASSES)
  # to 1-hot dense float vectors (that is we will have batch_size vectors,
  # each with NUM_CLASSES values, all of which are 0.0 except there will
  # be a 1.0 in the entry corresponding to the label).
  # 如3 -> [0,0,0,1,0,0,0,0,0,0]
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(1, [indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0) ## previously: tf.pack
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, # 比较inference()函数与1-hot标签所输出的logits Tensor
                                                          onehot_labels,
                                                          name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean') # 计算batch维度（第一维度）下交叉熵的平均值，将该值作为总损失
  return loss # 返回损失值的Tensor


def training(loss, learning_rate):
  """Sets up the training Ops.
    添加了通过梯度下降将损失最小化所需的操作
  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  # 从loss()函数中获取损失Tensor交给tf.summary.scalar，在与SummerWriter配合使用时可以向事件文件中生成汇总值，这里每次写入汇总值时都会释放损失Tensor的当前值
  tf.summary.scalar(loss.op.name, loss) ## previously: tf.scalar_summary
  # Create the gradient descent optimizer with the given learning rate.
  # 负责按照所要求的学习效率应用梯度下降法
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  # 保存全局训练步骤的数值
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  # 更新系统中的三角权重、增加全局步骤的操作，是session诱发一个完整训练步骤所必须运行的操作
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op # 返回包含了训练操作输出结果的Tensor


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  # 如果在K个最有可能的预测中可以发现真的标签，那么这个操作就会将模型输出标记为正确，这里把K的值设为1，即只有在预测为真的标签时才判断是正确的
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
