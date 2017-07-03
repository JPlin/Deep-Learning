# -*- coding: utf-8 -*-
#实现反向传播实现MINST的识别
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#定义神经网络的结构相关参数
INPUT_NODE = 784 #输入节点数
OUTPUT_NODE = 10    #输出的节点数
'''
通过改变下面这个参数来改变中间神经元的个数
'''
LAYER1_NODE = 10    #中间层1的节点数
LAYER2_NODE = 10    #中间层2的节点数
NUM_TEST = 1000 #一次测试所用的数据量
'''
通过改变NUM_TRAIN 来改变一次batch 所要用的数据
'''
NUM_TRAIN = 50 #一次训练一个batch 包含的数据量
TRAIN_TIMES = 10000 #训练的轮数

'''
通过改变eta 来改变学习率
'''
eta = tf.constant(0.5)

def get_weight_variable(shape,regularizer=None):
    #得到权值变量，封装变量的初始化过程
    #weights = tf.Variable(tf.truncated_normal(shape,stddev=1))
    weights = tf.Variable(tf.zeros(shape))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def get_biases_variable(shape , regularizer=None):
    #得到偏置变量，封装变量的初始化过程
    #biases = tf.Variable(tf.truncated_normal(shape,stddev=1)) 
    biases = tf.Variable(tf.zeros(shape))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(biases))
    return biases

def sigma(x):
    #实现 sigmoid 函数
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0),
                  tf.exp(tf.negative(x))))

def sigmaprime(x):
    #sigmoid 函数的求导
    return tf.multiply(sigma(x),tf.subtract(tf.constant(1.0),sigma(x)))

def variable_summaries(var,name):
    """在一个张量上面进行多种数据的汇总 (用于tensorboard 可视化)"""
    with tf.name_scope('summary_'+name):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean) #平均值
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev) #标准差
      tf.summary.scalar('max', tf.reduce_max(var)) #最大值
      tf.summary.scalar('min', tf.reduce_min(var)) #最小值
      tf.summary.histogram('histogram', var) #柱状图

#定义输入和输出        
x = tf.placeholder(tf.float32,[None,INPUT_NODE],name="x-input")
y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="y-input")

#***********************没有隐藏层 start**************
'''
#定义参数
w_1 = get_weight_variable([INPUT_NODE,OUTPUT_NODE])
b_1 = get_biases_variable([1,OUTPUT_NODE])
#前向传播
with tf.name_scope('layer1'):
    z_1 = tf.add(tf.matmul(x, w_1), b_1)
    y = sigma(z_1)
    variable_summaries(w_1,'w_1')
    variable_summaries(b_1,'b_1')
#反向求导
with tf.name_scope('detrivation'):
    delta_E = tf.subtract(y , y_)

    delta_z_1 = tf.multiply(delta_E,sigmaprime(z_1))
    delta_b_1 = delta_z_1
    delta_w_1 = tf.matmul(tf.transpose(x),delta_z_1)   
step = [
    tf.assign(w_1,
            tf.subtract(w_1, tf.multiply(eta, delta_w_1)))
  , tf.assign(b_1,
            tf.subtract(b_1, tf.multiply(eta,
                               tf.reduce_mean(delta_b_1, axis=[0]))))
]  
'''
#************************没有隐藏层 end**************

#***********************有一个隐藏层 start**************

#定义参数
w_1 = get_weight_variable([INPUT_NODE,LAYER1_NODE])
b_1 = get_biases_variable([1,LAYER1_NODE])
w_2 = get_weight_variable([LAYER1_NODE,OUTPUT_NODE])
b_2 = get_biases_variable([1,OUTPUT_NODE])
#前向传播
with tf.name_scope('layer1'):
    z_1 = tf.add(tf.matmul(x, w_1), b_1)
    y_1 = sigma(z_1)
    variable_summaries(w_1,'w_1')
    variable_summaries(b_1,'b_1')
with tf.name_scope('layer2'):
    z_2 = tf.add(tf.matmul(y_1, w_2), b_2)
    y = sigma(z_2)    
    variable_summaries(w_2,'w_2')
    variable_summaries(b_2,'b_2')
#计算残差（实际观察的值 与 估计的值 之间的差值）
#利用反向求导求出每一项对于最终结果的delta 值
with tf.name_scope('detrivation'):
    delta_E = tf.subtract(y , y_)

    delta_z_2 = tf.multiply(delta_E,sigmaprime(z_2))
    delta_b_2 = delta_z_2
    delta_w_2 = tf.matmul(tf.transpose(y_1),delta_z_2)

    delta_y_1 = tf.matmul(delta_z_2,tf.transpose(w_2))
    delta_z_1 = tf.multiply(delta_y_1,sigmaprime(z_1))
    delta_b_1 = delta_z_1
    delta_w_1 = tf.matmul(tf.transpose(x),delta_z_1)
#然后根据求导 更新参数，包括对 权值和偏置量 的更新,
step = [
    tf.assign(w_1,
            tf.subtract(w_1, tf.multiply(eta, delta_w_1)))
  , tf.assign(b_1,
            tf.subtract(b_1, tf.multiply(eta,
                               tf.reduce_mean(delta_b_1, axis=[0]))))
  , tf.assign(w_2,
            tf.subtract(w_2, tf.multiply(eta, delta_w_2)))
  , tf.assign(b_2,
            tf.subtract(b_2, tf.multiply(eta,
                               tf.reduce_mean(delta_b_2, axis=[0]))))
]

#***********************有一个隐藏层 end**************


#***********************有两个隐藏层 start**************'
'''
#定义参数
w_1 = get_weight_variable([INPUT_NODE,LAYER1_NODE])
b_1 = get_biases_variable([1,LAYER1_NODE])
w_2 = get_weight_variable([LAYER1_NODE,LAYER2_NODE])
b_2 = get_biases_variable([1,LAYER2_NODE])
w_3 = get_weight_variable([LAYER2_NODE,OUTPUT_NODE])
b_3 = get_biases_variable([1,OUTPUT_NODE])
#前向传播
with tf.name_scope('layer1'):
    z_1 = tf.add(tf.matmul(x, w_1), b_1)
    y_1 = sigma(z_1)
    variable_summaries(w_1,'w_1')
    variable_summaries(b_1,'b_1')
with tf.name_scope('layer2'):
    z_2 = tf.add(tf.matmul(y_1, w_2), b_2)
    y_2 = sigma(z_2)    
    variable_summaries(w_2,'w_2')
    variable_summaries(b_2,'b_2')
with tf.name_scope('layer3'):
    z_3 = tf.add(tf.matmul(y_2, w_3), b_3)
    y = sigma(z_3)    
    variable_summaries(w_3,'w_3')
    variable_summaries(b_3,'b_3')
#计算残差（实际观察的值 与 估计的值 之间的差值）
#利用反向求导求出每一项对于最终结果的delta 值
with tf.name_scope('detrivation'):
    delta_E = tf.subtract(y , y_)

    delta_z_3 = tf.multiply(delta_E,sigmaprime(z_3))
    delta_b_3 = delta_z_3
    delta_w_3 = tf.matmul(tf.transpose(y_2),delta_z_3)

    delta_y_2 = tf.matmul(delta_z_3,tf.transpose(w_3))
    delta_z_2 = tf.multiply(delta_y_2,sigmaprime(z_2))
    delta_b_2 = delta_z_2
    delta_w_2 = tf.matmul(tf.transpose(y_1),delta_z_2)

    delta_y_1 = tf.matmul(delta_z_2,tf.transpose(w_2))
    delta_z_1 = tf.multiply(delta_y_1,sigmaprime(z_1))
    delta_b_1 = delta_z_1
    delta_w_1 = tf.matmul(tf.transpose(x),delta_z_1)
#然后根据求导 更新参数，包括对 权值和偏置量 的更新,
step = [
    tf.assign(w_1,
            tf.subtract(w_1, tf.multiply(eta, delta_w_1)))
  , tf.assign(b_1,
            tf.subtract(b_1, tf.multiply(eta,
                               tf.reduce_mean(delta_b_1, axis=[0]))))
  , tf.assign(w_2,
            tf.subtract(w_2, tf.multiply(eta, delta_w_2)))
  , tf.assign(b_2,
            tf.subtract(b_2, tf.multiply(eta,
                               tf.reduce_mean(delta_b_2, axis=[0]))))
  , tf.assign(w_3,
            tf.subtract(w_3, tf.multiply(eta, delta_w_3)))
  , tf.assign(b_3,
            tf.subtract(b_3, tf.multiply(eta,
                               tf.reduce_mean(delta_b_3, axis=[0]))))
]
'''
#***********************有两个隐藏层 end**************

#开始训练和测试
#计算出测试时准确率
accuracy = tf.Variable(0)
accuracy_rate = tf.Variable(0)
accuracy = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#accuracy_rate = tf.reduce_sum(tf.cast(accuracy,tf.float32))/tf.cast(tf.constant(NUM_TEST),tf.float32)
accuracy_rate = tf.reduce_mean(tf.cast(accuracy,"float"))
tf.summary.scalar('accuracy_rate', accuracy_rate)    

sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('F:\Deep Learning\Learning\MNIST\log', sess.graph)
sess.run(tf.global_variables_initializer())
    
for i in range(TRAIN_TIMES):
    batch_xs , batch_ys= mnist.train.next_batch(NUM_TRAIN)
    acc,summary,_ = sess.run([accuracy_rate,merged,step],feed_dict={x:batch_xs,y_:batch_ys})
    writer.add_summary(summary,i)
    if i % 1000 == 0 :
        print(acc*100,'-%')
        res = sess.run(accuracy_rate,feed_dict={
                x:mnist.test.images[:NUM_TEST],y_:mnist.test.labels[:NUM_TEST]})
        print(res*100,'%')
        
writer.close()
sess.close()