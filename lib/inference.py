# -*- coding: utf-8 -*-
import tensorflow as tf

#定义神经网络的结构相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


#训练时创建，测试时通过加载变量取值
#训练时使用变量自身，在测试时使用变量的滑动平均值

#定义权值变量
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable(
            'weights',shape,
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
    
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
        
    return weights;

#定义偏置值
def get_biases_variable(shape , regularizer):
    biases = tf.get_variable(
            "biases",shape,
            initializer =tf.constant_initializer(0.0))
    
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(biases))
    return biases;

#定义前向传播过程
def inference(input_tensor , regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
                [INPUT_NODE,LAYER1_NODE],regularizer)
        biases = get_biases_variable(
                [LAYER1_NODE],regularizer)
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
        
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
                [LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = get_biases_variable(
                [OUTPUT_NODE],regularizer)
        layer2 = tf.nn.relu(tf.matmul(layer1,weights) + biases)
        
    return layer2


        