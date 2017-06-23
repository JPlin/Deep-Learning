# -*- coding: utf-8 -*-
#-coding:utf-8-

'''
导入MNIST 数据集
    包含有60000 行的训练数据集，mnist.train 
    还有10000 行的测试数据集,mnist.test

有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，
从而更加容易把设计的模型推广到其他数据集上

图片设置为 xs ,标签设置为 ys ,图片为 28X28 = 784 pi

所以 mnist.train.images 是一个 [60000,784] 的张量
所以 mnist.train.labels 是一个 [60000,10]  的张量

softmax可以看成是一个激活（activation）函数或者链接（link）函数
把我们定义的线性函数的输出转换成我们想要的格式

'''
#导入数据
import tensorflow.examples.tutorials.mnist.input_data as id
mnist = id.read_data_sets("MNIST_data/", one_hot=True)
import matplotlib.pyplot as plt
#导入tensorflow
import tensorflow as tf


#预定义变量来存储 图片的数据,是一个占位符，运行时填充
#x 为图片像素灰度化的数据
#y 为图片代表的正确标签

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder("float",[None,10])

#variable 可以修改的张量,W 是权值张量，b 是偏移张量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
corrent_prediction_ing = tf.Variable(0)
accuracy_ing = tf.Variable(0)

'''
  使用激活函数用来产生非线性的结果
  有不同的激活函数可以选择
  本次实验使用 softmax 和 sigmoid 分别进行实验并对结果进行分析比对
'''
y = tf.nn.softmax(tf.matmul(x,W)+b)
#y = tf.nn.sigmoid(tf.matmul(x,W)+b)

'''
  损失函数用来评估一个模型的好坏，应该尽量减小损失函数的数值
  有不同的损失函数可以选择
  本次实验使用 交叉熵(cross_entropy) 和 方差(cost) 分别进行实验并对结果进行比对
''' 
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cost = tf.pow(y-y_,2)/2


#使用梯度下降算法 以 0.01的速率最小化学习的 交叉熵 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


correct_prediction_ing = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy_ing = tf.reduce_mean(tf.cast(correct_prediction_ing,"float"))
'''
上面已经定义好了模型
tensorflow 有一张数据流图，描述了各个计算单元
首先 可以自动运用反向传播的算法确定你的变量是怎么影响最小化的成本值
然后 tensorflow选择优化算法不断修改变量以 降低成本
最后 只返回一个单一的操作,执行这个操作，tensorflow 会帮你做很多事情
下面开始训练
'''
#变量初始化 
init  = tf.initialize_all_variables()
#在一个会话里面启动模型
sess = tf.Session()
sess.run(init)

#对模型训练1000次
#随机选取100个图像数据，进行随机训练
li = []
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    li.append(sess.run(accuracy_ing,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
'''
训练完成
现在对模型的性能进行评估
'''
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


#制作出一个图表
plt.figure(figsize=(13,7)) #创建绘图对象 
plt.plot(li,"b--",label='Original data')
plt.title("Oh my god")
plt.ylabel("accuracy")
#
plt.savefig("images/softmax_cost.png")
plt.show()
plt.close()
'''
疑问：
    训练1000 和训练 5000 差别很小，说明算法有局限性，不能单单通过训练的次数来提高
'''