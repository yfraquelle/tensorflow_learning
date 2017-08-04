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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import tensorflow.python.platform
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
from tensorflow.examples.tutorials.mnist import mnist


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data/', 'Directory to put the training data.') ## previously: data
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  # 查询给定的DataSet，索要下一批次batch_size的图像和标签，与占位符相匹配的Tensor包含下一批次的图像和标签
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  # 以占位符为哈希键，创建一个Python字典对象，键值是其代表的反馈Tensor
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  } # 这个字典随后作为feed_dict参数传入sess.run()函数中，为这一步的训练提供输入样例
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  # 创建一个循环
  for step in xrange(steps_per_epoch):
    # 添加feed_dict
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    # 在调用sess.run()时传入eval_correct操作，用给定的数据集评估模型
    # true_count 变量会累加所有 in_top_k 操作判定为正确的预测之和。接下来，只需要将正确测试的总数，除以例子总数，就可以得出准确率了。
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  # 确保本地训练文件夹中已经下载了正确的数据，然后将这些数据解压并返回一个含有DataSet实例的字典
  # data_sets.train 55000个图像和标签，作为主要训练集
  # data_sets.validation 5000个图像和标签，用于迭代验证训练准确度
  # data_sets.test 10000个图像和标签，用于最终测试训练准确度
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  # 在打开默认图表（Graph）之前，我们应该先调用 get_data(train=False) 函数，抓取测试数据集。
  # test_all_images, test_all_labels = get_data(train=False)
  with tf.Graph().as_default(): # 表示所有yi'ji个构建的操作都要与默认的tf.Graph全局实例关联起来，它是一系列可以作为整体执行的操作，大部分场景只需要以来默认图表的一个实例即可
    ## 为数据创建占位符
    # Generate placeholders for the images and labels.定义传入图表中的shape参数，包括batch_size值，在训练循环的后续步骤中，传入的整个图像和标签数据集会被切片以符合每一个操作所设置的batch_size值，然后使用feed_dict参数，将数据传入sess.run()函数
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    ## 运行mnist.py文件，构建图表
    # Build a Graph that computes predictions from the inference model.返回包含了预测结果的Tensor，促使神经网络向前反馈并做出预测的要求
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2) # 接受图像占位符作为输入，在此基础上借助ReLU激活函数，构建一对完全连接层和一个有着10个节点、指明了输出logtis模型的线性层

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # 推理
    # Add the Op to compare the logits to the labels during evaluation.
    # 在进入训练循环之前，先调用，传入的logits和标签参数要与loss函数的一致，这是为了先构建Eval操作
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    # 为了释放TensorBoard所使用的事件文件，所有的即时数据都要在图表构建阶段合并至一个操作中
    summary_op =  tf.summary.merge_all()## previously: tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    # 保存用来后续恢复模型以进一步训练或评估的检查点文件
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    # 可以使用with代码块生成Session限制作用域 with tf.Session() as sess:
    sess = tf.Session() # 没有传入参数，表明该代码将会依附于默认的本地会话

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init) # sess.run()将会运行图表中与作为参数传入的操作相对应的完整子集，在初次调用时，init操作只包含了变量初始化程序tf.group，图表的其他部分在下面的训练循环运行

    # Instantiate a SummaryWriter to output summaries and the Graph.
    # 写入包含了图表本身和即时数据具体值的事件文件
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)## previously: tf.train.SummaryWriter(

    # And then after everything is built, start the training loop.
    # 由于train_op不会产生输出，在返回的元组中的对应元素就是None，所以会被抛弃，但是如果模型在孙连中出现偏差，loss Tensor的值可能会变成NaN，所以要获取它的值并记录下来
    # 假设训练一切正常，没有出现NaN，训练循环灰每隔100个训练步骤，就打印一行状态文本
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      # 生成一个反馈字典，其中包含对应步骤中训练所要使用的例子，他们的哈希键就是其代表的占位符操作
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss], # 要获取[train_op,loss]，sess.run()会返回一个有两个元素的元组，其中每一个Tensor对象，对应了返回的元组中的numpy数组，而这些数组中包含了当前这步训练中对应Tensor的值
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        # 每次运行summary_op时，都会往事件文件中写入最新的即时数据，函数的输出会传入事件文件读写器的add_summary函数
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        # 定义调用，向训练文件夹中写入包含了当前所有可训练变量值得检查点文件，这以后就可以使用saver.restore()方法，重载模型的参数继续训练：saver.restore(sess,FLAGS.train_dir)
        saver.save(sess, FLAGS.train_dir, global_step=step)
        # 尝试使用训练数据集与测试数据集对模型进行评估，更复杂的使用场景通常是先隔绝data_sets.test测试数据集，只有在大量的超参数优化调整之后才进行检查
        # Evaluate against the training set.训练数据集
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.验证数据集
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.测试数据集
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
