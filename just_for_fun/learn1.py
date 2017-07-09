#coding:utf-8
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1000;
vectors_set = []
for i in range(num_points):
	#产生正太分布的数据
	x1 = np.random.normal(0.0,0.55)
	y1 = x1*0.1 +0.3 + np.random.normal(0.0,0.03)
	vectors_set.append([x1,y1])

#x 和 y 的值分别取出来
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

#制作出一个图表
plt.plot(x_data,y_data,'ro',label='Original data')
plt.legend()
plt.show()

#variable 方法定义的变量会保存在内部图数据结构中
w = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = w*x_data + b

#定义一个cost function 来确保迭代的结果正在变好

loss = tf.reduce_mean(tf.square(y-y_data))
#使用梯度下降算法来最小化损失函数
#梯度就像一个指南针，朝着最小的方向前进
#为了计算梯度，tensorflow 会对错误函数求导
#算法需要在w 和 b 计算部分导数，以在每次迭代中指明方向
#0.5 速率，是一个步长，不能太大也不能太小
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(8):
	sess.run(train)
	print (step,sess.run(w),sess.run(b))
	print (step,sess.run(loss))

	plt.plot(x_data,y_data,'ro')
	plt.plot(x_data,sess.run(w) * x_data + sess.run(b))
	plt.legend()
	plt.show()


