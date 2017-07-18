# -*- coding: utf-8 -*-
#利用RNN 中的LSTM 模型进行MNIST 手写数字的识别
import tensorflow as tf  
import sys  
from tensorflow.examples.tutorials.mnist import input_data  
  
# this is data  
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  
  
# hyperparameters  
lr = 0.001  #学习率
training_iters = 100000  
batch_size = 128  #一个batch 的大小
  
n_inputs = 28   # MNIST data input (img shape: 28*28)  
n_steps = 28    # time steps,一个图片的列数
n_hidden_units = 128   # neurons in hidden layer  
n_classes = 10      # MNIST classes (0-9 digits)  
  
# tf Graph input  
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  
y = tf.placeholder(tf.float32, [None, n_classes])  
  
# Define weights  
weights = {  
    # (28, 128)  
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),  
    # (128, 10)  
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))  
}  
biases = {  
    # (128, )  
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  
    # (10, )  
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))  
}  
  
  
def RNN(X, weights, biases):  
    # hidden layer for input to cell  
    ########################################  
    #X(128 batch,28 steps,28 inputs)  
    #==>(128*28,28 inputs)  
    X = tf.reshape(X,[-1,n_inputs])      
    #==>(128 batch*28 steps,128 hidden)  
    X_in = tf.matmul(X,weights['in'])+biases['in']  
    #==>(128 batch,28 steps,128 hidden)  
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])  
    # cell  
    ##########################################  
    
    #same to define active function  
    # 定义一个 LSTM 结构，LSTM 中使用的变量会在该函数中自动被声明
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)  
    #lstm cell is divided into two parts(c_state,m_state)  
    # 将 LSTM 中的状态初始化为全 0  数组，batch_size 给出一个 batch 的大小
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)  
      
    #choose rnn how to work,lstm just is one kind of rnn,use lstm_cell for active function,set initial_state
    #用来构建一个循环的 RNN
    #args：
    #       cell = RNN 的神经单元
    #       inputs = 数据输入
    #       initial_state = 初始的循环神经结构的状态全0
    #returns:
    #outputs 是每个时刻的输出的序列
    #states 是最后的状态的输出
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)     
    
    # hidden layer for output as the final results  
    #############################################  
    results = tf.matmul(states[1],weights['out']) + biases['out']     
      
    #unpack to list [(batch,outputs)]*steps  
    #outputs = tf.unpack(tf.transpose(outputs,[1,0,2])) # state is the last outputs  
    #results = tf.matmul(outputs[-1],weights['out']) + biases['out']  
    return results,states  
  
  
pred,states = RNN(x, weights, biases)  

#训练
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  
tf.summary.scalar('loss', cost) 
train_op = tf.train.AdamOptimizer(lr).minimize(cost)  
  
#测试
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
tf.summary.scalar('accuracy', accuracy)  
  
sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('log/', sess.graph)
 
sess.run(tf.global_variables_initializer())
step = 0  
while step * batch_size < training_iters:  
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)  
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])  
    _,state1,state2,summary = sess.run([train_op,states[0],states[1],merged], feed_dict={  
        x: batch_xs,  
        y: batch_ys,  
    })  
    if step % 50 == 0: 
       batch_testx ,batch_testy = mnist.test.next_batch(batch_size)
       batch_testx = batch_testx.reshape([batch_size, n_steps, n_inputs])
       acc =sess.run([accuracy], feed_dict={  
       x: batch_testx,  
       y: batch_testy,  
       }) 
       print(acc)
       writer.add_summary(summary,step)
    if step % 1000 ==0:
       print(state1 , state2)
    step += 1  
writer.close()
sess.close()
