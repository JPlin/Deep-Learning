# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import inference

#配置神经网络的参数
BATCH_SIZE = 100            #一个训练batch 中的数据个数
LEARNING_RATE_BASE = 0.8    #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
REGULARAZTION_RATE = 0.0001 #正则化项在损失函数中的系数
TRAINING_STEPS = 30000      #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均值的衰减率
TIMES_OF_TRAIN = 100        #训练完一遍数据所需要的轮数

MODEL_SAVE_PATH = "/model"
MODEL_NAME="model.ckpt"

def train(input):
    
    x = tf.placeholder(
            tf.float32,[None,inference.INPUT_NODE],name="x-input")
    y_ = tf.placeholder(
            tf.float32,[None,inference.OUTPUT_NODE],name="y-input")
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    #前向传播的结果
    y = inference(x,regularizer)
    
    #存储训练轮数的变量，无需计算滑动平均值
    global_step = tf.Variable(0,trainable=False)
    
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY,global_step)
    variable_average_op = variable_averages.apply(tf.trainable_variables())
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            y,tf.arg_max(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            TIMES_OF_TRAIN,
            LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss,global_step = global_step)
    
    with tf.control_dependencies([train_step,variable_average_op]):
        train_op = tf.no_op(name="train")
        
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        for i in range(TRAINING_STEPS):
            xs,ys = None,None
            _,loss_value ,step = sess.run([train_op , loss, global_step],feed_dict={x:xs,y_:ys})
            
            if i%1000 == 0:
                saver.save(sess,
                           os.path.join(MODEL_SAVE_PATH,MODEL_NAME)
                           ,global_step = global_step)
    
def main(argv = None):
    #input = xxx
    train(input)
     
if __name__ == '__main__':
    tf.app.run()