# -*- coding: utf-8 -*-
#导入数据
import tensorflow.examples.tutorials.mnist.input_data as id
mnist = id.read_data_sets("MNIST_data", one_hot=True)
#导入tensorflow
import tensorflow as tf

#定义神经网络的结构相关参数
INPUT_NODE = 784 #输入节点数
OUTPUT_NODE = 10 #输出的节点数
NUM_TRAIN = 50 #一次训练一个batch 包含的数据量
TRAIN_TIMES = 5000 #训练的轮数
LEARNING_RATE = 1e-4 #学习率
DROP_OUT=0.5 #防止过拟合的失效参数

def get_weight_variable(shape,regularizer=None):
    #得到权值变量，封装变量的初始化过程
    weights = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def get_biases_variable(shape , regularizer=None):
    #得到偏置变量，封装变量的初始化过程
    biases = tf.Variable(tf.constant(0.1,shape=shape)) 
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(biases))
    return biases

def variable_summaries(var,name):
    #在一个张量上面进行多种数据的汇总 (用于tensorboard 可视化)
    with tf.name_scope('summary_'+name):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean) #平均值
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev) #标准差
      tf.summary.scalar('max', tf.reduce_max(var)) #最大值
      tf.summary.scalar('min', tf.reduce_min(var)) #最小值
      tf.summary.histogram('histogram', var) #柱状图    
def conv2d(x, W):
    #卷积，步长为1，
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #池化，采用最大值池化，ksize代表2X2作为池化模版大小，第一和第四必须为1
    #strides 步长，第一维和最后一维只能为1，
    #VALID 代表不使用全0填充，SAME 代表使用全0填充
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def avg_pool_2x2(x):
    #另一种池化，平均池化
    return tf.nn.avg_pool(x,ksize=[1,2,2,1],
                         strides=[1,2,2,1],padding="SAME")

#定义输入和输出        
x = tf.placeholder(tf.float32,[None,INPUT_NODE],name="x-input")
y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="y-input")

#********************第一层卷积*******************
with tf.name_scope("Convolution_1"):
    #patch 大小是5X5
    #输入通道是1，输出通道是32
    W_conv1 = get_weight_variable([5, 5, 1, 32])
    #每一个输出通道对应一个偏置量
    b_conv1 = get_biases_variable([32])
    #把x变成4D 的向量，2,3维分别代表图片的高宽，最后一维代表颜色通道数
    x_image = tf.reshape(x,[-1,28,28,1])
    #进行卷积，加上偏置项，应用ReLU 激活函数，进行max_pooling
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

#********************第二层卷积*******************

with tf.name_scope("Convolution_2"):
    #patch 大小是5x5
    #输入通道是32，输出通道是64
    W_conv2 = get_weight_variable([5, 5, 32, 64])
    b_conv2 = get_biases_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
   
#********************第三层卷积*******************
'''
with tf.name_scope("Convolution_3"):
    #patch 大小是5x5
    #输入通道是32，输出通道是64
    W_conv3 = get_weight_variable([5, 5, 64, 64])
    b_conv3 = get_biases_variable([64])
    
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
'''
#********************全链接层*********************

with tf.name_scope("Full_con"):
    #h_pool2 的结果图片的尺寸为 7x7
    #加入一个拥有 1024 个神经元的 全链接层
    W_fc1 = get_weight_variable([7*7*64,1024])
    b_fc1 = get_biases_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#********************Dropout********************

    #使用dropout 为了减少过拟合
    #通常dropout保持不变，并且在训练的过程中启用，在测试的时候关闭
keep_prob = tf.placeholder("float")
y_conv_drop = tf.nn.dropout(h_fc1,keep_prob)

#********************输出层********************
with tf.name_scope("Output"):
    #与一般的mnist一样，有10个神经元，
    #使用softmax 进行不同结果的概率计算
    W_fc2 = get_weight_variable([1024,10])
    b_fc2 = get_biases_variable([10])
    
    y_conv = tf.nn.softmax(tf.matmul(y_conv_drop,W_fc2)+b_fc2)
    

#********************开始训练*******************
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
tf.summary.scalar('loss', cross_entropy) 
#使用更加复杂的Adam 梯度下降算法
step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy) 
#interactivesession 它能让你在运行图的时候，插入一些计算图，
#这些计算图是由某些操作(operations)构成的。
sess = tf.InteractiveSession()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('log', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(TRAIN_TIMES):
    batch = mnist.train.next_batch(NUM_TRAIN)
    sess.run(step,feed_dict={x: batch[0], y_: batch[1], keep_prob: DROP_OUT})
    summary,acc = sess.run([merged,accuracy] ,feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob: 1.0})
    writer.add_summary(summary,i)
    if i%500 == 0:
        acc = sess.run(accuracy ,feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob: 1.0})
        print(acc)
    
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
writer.close()
sess.close()