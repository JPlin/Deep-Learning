
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf


def _read_words(filename):
    #获得最原始的单词数据
  with tf.gfile.GFile(filename, "r") as f: #通过gfile api 访问文件系统
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    # 构建一个带有索引的单词表
  data = _read_words(filename)

  counter = collections.Counter(data) # 集合的计数器类，键值对存储
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))  # 先按出现的次数排序，再按字母顺序排序

  words, _ = list(zip(*count_pairs))  #解压缩的过程，将次数和单词分隔开，只保留单词顺序
  word_to_id = dict(zip(words, range(len(words))))  # 按序给单词赋予一个唯一的id

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
    #根据文件的单词，找出对应的id
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    #生成各个数据集中的单词的id ， 以及词汇表的长度
    #Returns:
    #tuple (train_data, valid_data, test_data, vocabulary)
    #where each of the data objects can be passed to PTBIterator.

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary

def ptb_id_to_word(id,data_path="./simple-examples/data"):
  train_path = os.path.join(data_path, "ptb.train.txt")
  word_to_id = _build_vocab(train_path)
  id_to_word = {v:k for k,v in word_to_id.items()}
  return id_to_word[id]

def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)  #转换成tensor

    data_len = tf.size(raw_data)   # 总的单词的个数
    batch_len = data_len // batch_size # batch_len ？
                                      # batch_size 每批数据的句子的个数
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps # epoch_size 总共有多少批
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    # tf.control_dependencies函数，用于先执行assertion操作，再执行当前context中的命令
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    # tf.train.range_input_producer产生一个从1至epoch_size-1的整数至队列中
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    #第一个参数，为切割的起点
    #第二个参数，为切割的终点
    #可以看出，label y是训练样本x的下一个词
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

if __name__ == "__main__":
    for a in [10,12,45]: print(ptb_id_to_word(a))