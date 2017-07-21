# -*- coding: utf-8 -*-
"""
Example / benchmark for building a PTB LSTM model
model described in :http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
"""
#导入feature 是为了使用新版本python 3.x 的特性，如启用相对导入，应用新的除法语法，新的print函数
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time

import numpy as np
import tensorflow as tf

import reader

'''start of 构建全局参数'''
flags = tf.flags
logging = tf.logging

flags.DEFINE_string( # 定义变量 model 为 small ，后面的为注释
    "model", 
    "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string( # 定义下载好的数据的存放位置
    "data_path", 
    "F:\Deep Learning\Learning\RNN\simple-examples\data",
    "Where the training/test data is stored.")
flags.DEFINE_string( # log 文件的保存位置
    "save_path", 
    "log/",
    "Model output directory.")
flags.DEFINE_bool( # 是否使用 float16格式
    "use_fp16", 
    False,
    "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
'''end of 构建全局参数'''

def data_type(): #返回使用的数据类型
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    #:param is_training: 是否要进行训练.如果is_training=False,则不会进行参数的修正。
    self._input = input_

    batch_size = input_.batch_size  # 每批数据有几个序列
    num_steps = input_.num_steps  # 序列的长度
    size = config.hidden_size # 隐藏层中神经元的数目
    vocab_size = config.vocab_size # 词典的规模大小

    # 使用forget_bias 结果会略好一点
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)

    attn_cell = lstm_cell
    # 网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。
    # 这是是一种有效的正则化方法，可以有效防止过拟合
    # 同一个t时刻中，多层cell之间传递信息的时候进行dropout    
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    # 使用多层的LSTM网络结构
    # config.num_layers ：层数
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
    self._initial_state = cell.zero_state(batch_size, data_type())

    # 输入预处理
    with tf.device("/cpu:0"): #使用0号gpu 进行操作
      embedding = tf.get_variable(
          # vocab size * hidden size, 将单词转成embedding描述
          "embedding", [vocab_size, size], dtype=data_type())
      # 从embedding 中查找输入单词序列的索引所对应的词向量
      # 将输入seq用embedding表示, shape=[batch_size*num_steps, hidden_size]
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # 对于tensorflow中的rnn() 函数的一种简单实现
    # 创建了一个没有的LSTM为了便于理解，和于tutorial 保持一致
    # 一般情况下使用rnn() 或者 state_saving_rnn() 
    #
    # The alternative version of the code below is:
    #
    # inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state # state 表示 各个batch中的状态
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        # cell_out: [batch_size, hidden_size]
        # 按顺序将数据序列输入即可，状态会自动保存
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output) # output: shape[num_steps][batch_size,hidden_size]

    # 把之前的list展开，成[batch_size, hidden_size*num_steps],然后 reshape 成[batch_size*numsteps, hidden_size]
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])

    # softmax_w , shape=[hidden_size, vocab_size], 用于将distributed表示的单词转化为one-hot表示    
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    # [batch_size*num_steps, vocab_size] 从隐藏语义转化成完全表示
    logits = tf.matmul(output, softmax_w) + softmax_b

    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

    # use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits, # 预测结果分布,shape=[batch_size,num_steps,vocab_size]
        input_.targets, # 正确的结果值 shape=[batch_size,num_steps]
        tf.ones([batch_size, num_steps], dtype=data_type()), # 权值
        average_across_timesteps=False, # 是否对不同的时刻之间求平均
        average_across_batch=True # 是否对不同批数据之间求平均
    ) #return shape=[num_steps]

    # update the cost variables
    self._cost = cost = tf.reduce_sum(loss)
    self._final_state = state

    # 没有训练就可以直接返回，已经得出了损失值和最后的状态
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    # tf.trainable_variables 可以得到整个模型中所有trainable=True的Variable。
    # 实际得到的tvars是一个列表，里面存有所有可以进行训练的变量。
    tvars = tf.trainable_variables()
   
    # clip_by_global_norm: 梯度衰减，具体算法为t_list[i] * clip_norm / max(global_norm, clip_norm)
    # 这里gradients求导，ys和xs都是张量
    # 返回一个长为len(xs)的张量，其中的每个元素都是\grad{\frac{dy}{dx}}
    # clip_by_global_norm 用于控制梯度膨胀,前两个参数t_list, global_norm, 则
    # t_list[i] * clip_norm / max(global_norm, clip_norm)
    # 其中 global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
    #
    # 形象解释：当待裁剪的梯度的张量的模大于指定的大小时，会做等比例缩放
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)

    # 梯度下降优化器，指定学习速率
    # 在运行过程中想要调整gradient值，就不能直接简单的optimizer.minimize(loss)而是要显式计算gradients
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(  # 将梯度应用于变量，不直接使用minimize
        zip(grads, tvars),
        #优化器内置的global_step 用于记录全局的训练次数，最终的训练次数为max_max_epoch_size*max_epoch_size
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    # 使用 session 来调用 lr_update 操作
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0 # 学习速率
  max_grad_norm = 5 # 用于控制梯度膨胀
  num_layers = 2  # lstm的cell层数
  num_steps = 20  # 单个数据中，序列的长度
  hidden_size = 200 # 隐藏层规模
  max_epoch = 4 # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
  max_max_epoch = 13  # 指的是整个文本循环13遍
  keep_prob = 1.0 # 神经单元的保持率，防止过拟合
  lr_decay = 0.5  # 学习速率衰减
  batch_size = 20 # 每批数据的规模，每批有20个
  vocab_size = 10000  # 词典规模，总共10K个词


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  # epoch_size 表示批次总数。也就是说，需要向session喂这么多次数据
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c   # 这部分有什么用？看不懂
      feed_dict[h] = state[i].h

  if eval_op is None:
    output = session.run(model.input.targets, feed_dict)
    for x in output:print(reader.ptb_id_to_word(x))
    
    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10: # // 表示整数除法
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size,   # 训练的轮数 
             np.exp(costs / iters), # 平均损失
             iters * model.input.batch_size / (time.time() - start_time)))  # 每秒训练的词数

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path) #获取全部的三个数据集
  train_data, valid_data, test_data, _ = raw_data

  config = get_config() # 获取训练时的参数配置
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1  #获取测试时的参数配置

  with tf.Graph().as_default(): # 将一个图设置为默认图，自动设置全局节点加入一个默认图中
    initializer = tf.random_uniform_initializer(  # 定义一个初始化器，如何对参数变量初始化
      -config.init_scale,config.init_scale)

    with tf.name_scope("Train"):  # 训练模型， is_trainable=True
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
          m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #  交叉检验和测试模型，is_trainable=False
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    # supervisor
    # 自动去checkpoint加载数据或初始化数据 
    # 自身有一个Saver，可以用来保存checkpoint 
    # 有一个summary_computed用来保存Summary 
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)  #logdir用来保存checkpoint和summary
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)  # 计算衰减率
        m.assign_lr(session, config.learning_rate * lr_decay) # 更新学习率

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)  # 训练困惑度
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid) # 检验困惑度,tf.no_op() 不进行优化操作
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)  # 测试最终的困惑度,tf.no_op() 不进行优化操作
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step) #global_step 全局的步数


if __name__ == "__main__":
  tf.app.run()