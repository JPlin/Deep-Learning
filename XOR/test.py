import math
import tensorflow as tf
import numpy as np

HIDDEN_NODES_1 = 4
HIDDEN_NODES_2 = 4

x_input = tf.placeholder(tf.float32,shape=[None,2],name="x-input")
y_input = tf.placeholder(tf.float32,shape=[None,1],name="y-input")

W_hidden_1 = tf.Variable(tf.truncated_normal([2,HIDDEN_NODES_1],stddev=1./math.sqrt(2)))
b_hidden_1 = tf.Variable(tf.zeros([HIDDEN_NODES_1]))
hidden_1 = tf.nn.sigmoid(tf.matmul(x_input,W_hidden_1)+b_hidden_1)

W_hidden_2 = tf.Variable(tf.truncated_normal([HIDDEN_NODES_1,HIDDEN_NODES_2],stddev=1./math.sqrt(HIDDEN_NODES_1)))
b_hidden_2 = tf.Variable(tf.zeros(HIDDEN_NODES_2))
hidden_2 = tf.nn.sigmoid(tf.matmul(hidden_1,W_hidden_2)+b_hidden_2)

W_output = tf.Variable(tf.truncated_normal([HIDDEN_NODES_2,1],stddev=1./math.sqrt(HIDDEN_NODES_2)))
b_output = tf.Variable(tf.zeros(1))
y = tf.nn.sigmoid(tf.matmul(hidden_2,W_output)+b_output)

#cross_entropy = -tf.reduce_sum(y_input*tf.log(y))
cost = tf.pow(y_input-y,2)/2

# 训练步骤定义,在最小化交叉熵的情况下,以每次0.01的梯度调整参数
train_step = tf.train.AdamOptimizer(0.01).minimize(cost)


# 初始化之前定义好的全部变量
init = tf.global_variables_initializer()

# 定义会话并启动会话
sess = tf.Session()
sess.run(init)

xTrain = np.array([[0,0],[0,1],[1,0],[1,1]])
yTrain = np.array([[1],[0],[0],[1]])

for i in range(500):
    sess.run(train_step,feed_dict={x_input: xTrain, y_input: yTrain})
    if i % 10 == 0:
        #print("Step:", i, "Current loss:", loss_val)
        for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            print(sess.run(y, feed_dict={x_input:[x]}))