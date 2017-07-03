# -*- coding: utf-8 -*-
#导入数据
import tensorflow.examples.tutorials.mnist.input_data as id
mnist = id.read_data_sets("MNIST_data/", one_hot=True)
import matplotlib.pyplot as plt
#导入tensorflow
import tensorflow as tf

#定义神经网络的结构相关参数
INPUT_NODE = 784 #输入节点数
OUTPUT_NODE = 10    #输出的节点数
'''
通过改变下面参数来改变中间神经元的个数
'''
LAYER1_NODE = 30    #中间层的节点数
LAYER2_NODE = 30
NUM_TEST = 1000 #一次测试所用的数据量
'''
通过改变NUM_TRAIN 来改变一次batch 所要用的数据
'''
NUM_TRAIN = 10 #一次训练一个batch 包含的数据量
TRAIN_TIMES = 10000 #训练的轮数
'''
通过改变eta 来改变学习率
'''
eta = tf.constant(0.01)


def get_weight_variable(shape,regularizer=None):
    #得到权值变量，封装变量的初始化过程
    weights = tf.Variable(tf.truncated_normal(shape,stddev=1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def get_biases_variable(shape , regularizer=None):
    #得到偏置变量，封装变量的初始化过程
    biases = tf.Variable(tf.truncated_normal(shape,stddev=1)) 
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(biases))
    return biases
def variable_summaries(var,name):
    """在一个张量上面进行多种数据的汇总 (用于tensorboard 可视化)"""
    with tf.name_scope('summary_'+name):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

x = tf.placeholder(tf.float32,[None,INPUT_NODE])
y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE])

#***********************没有隐藏层 start**************
W = get_weight_variable([INPUT_NODE,OUTPUT_NODE])
b = get_biases_variable([OUTPUT_NODE])
#y = tf.nn.softmax(tf.matmul(x,W)+b)
y = tf.nn.sigmoid(tf.matmul(x,W)+b)
variable_summaries(W,"W")
variable_summaries(b,"b")
#***********************没有隐藏层 end**************

#***********************有一个隐藏层 start**************
'''
W1 = get_weight_variable([INPUT_NODE,LAYER1_NODE])
b1 = get_biases_variable([LAYER1_NODE])
y1 = tf.nn.sigmoid(tf.matmul(x,W1)+b1)
W2 = get_weight_variable([LAYER1_NODE,OUTPUT_NODE])
b2 = get_biases_variable([OUTPUT_NODE])
y  = tf.nn.sigmoid(tf.matmul(y,W2)+b2)
variable_summaries(W,"W1")
variable_summaries(b,"b1")
variable_summaries(W,"W2")
variable_summaries(b,"b2")
'''
#***********************有一个隐藏层 end**************

#***********************有两个隐藏层 start**************
'''
W1 = get_weight_variable([INPUT_NODE,LAYER1_NODE])
b1 = get_biases_variable([LAYER1_NODE])
y1 = tf.nn.sigmoid(tf.matmul(x,W1)+b1)
W2 = get_weight_variable([LAYER1_NODE,LAYER2_NODE])
b2 = get_biases_variable([LAYER2_NODE])
y2 = tf.nn.sigmoid(tf.matmul(y1,W2)+b2)
W3 = get_weight_variable([LAYER2_NODE,OUTPUT_NODE])
b3 = get_biases_variable([OUTPUT_NODE])
y  = tf.nn.sigmoid(tf.matmul(y2,W3)+b3)
variable_summaries(W1,"W1")
variable_summaries(b1,"b1")
variable_summaries(W2,"W2")
variable_summaries(b2,"b2")
variable_summaries(W3,"W3")
variable_summaries(b3,"b3")
'''
#***********************有两个隐藏层 end**************

'''
  损失函数用来评估一个模型的好坏，应该尽量减小损失函数的数值
''' 
accuracy = tf.Variable(0)
accuracy_rate = tf.Variable(0)
cost = tf.pow(y-y_,2)/2
#tf.summary.scalar('cost',cost)
#使用梯度下降算法 以 0.01的速率最小化学习的 交叉熵 
step = tf.train.AdamOptimizer(eta).minimize(cost)

accuracy = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy_rate = tf.reduce_mean(tf.cast(accuracy,"float"))
tf.summary.scalar('accuracy_rate', accuracy_rate)
'''
上面已经定义好了模型
下面开始训练
'''
sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('F:\Deep Learning\Learning\MNIST\log', sess.graph)
sess.run(tf.global_variables_initializer())

#对模型训练1000次
#随机选取100个图像数据，进行随机训练
li = []
for i in range(TRAIN_TIMES):
    batch_xs,batch_ys = mnist.train.next_batch(NUM_TRAIN)
    summary,_,acc = sess.run([merged,step,accuracy_rate],feed_dict={x:batch_xs,y_:batch_ys})
    writer.add_summary(summary,i)
    if i%1000 == 0:
        print(acc)
    li.append(sess.run(accuracy_rate,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
'''
训练完成
现在对模型的性能进行评估
'''
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


writer.close()
#制作出一个图表
plt.figure(figsize=(13,7)) #创建绘图对象 
plt.plot(li,"b--",label='Original data')
plt.title("ten_size_of_batch")
plt.ylabel("accuracy")
#
plt.savefig("images/ten_size_of_batch.png")
plt.show()
plt.close()