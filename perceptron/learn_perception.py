# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.io as sio 

def learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init=None,w_gen_fea=None):
    num_neg_examples = tf.size(neg_examples_nobias[0])
    num_pos_examples = tf.size(pos_examples_nobias[0])
    
    num_err_history = tf.Variable([1],name="err_history")
    #w_dist_history = []
    
    neg_examples = tf.concat((neg_examples_nobias,tf.ones([num_neg_examples,1])),1)
    pos_examples = tf.concat((pos_examples_nobias,tf.ones([num_pos_examples,1])),1)
    
    if w_init == None :
        w = tf.Variable(tf.truncated_normal([3,1]),name="W")
    else:
        w = w_init
    
    if w_gen_fea == None:
        w_gen_fea = tf.Varible([1],name="w_gen_fea")
        
    iterator = 0
    mistakes0, mistakes1 = eval_perceptron(neg_examples,pos_examples,w)
    num_errs = len(mistakes0) + len(mistakes1) 
    num_err_history.append(num_errs)
   
    print("Number of errors in iteration %d:%d\n"%(iterator,num_errs))
    print("weights\n",w)
    
    while len(num_errs)>0:
        iterator +=1
        
        w = update_weights(neg_examples,pos_examples,w)
        #if (length(w_gen_feas) ~= 0)
        #w_dist_history(end+1) = norm(w - w_gen_feas);
        #end
        mistakes0, mistakes1 = eval_perceptron(neg_examples,pos_examples,w)
        num_errs = len(mistakes0) + len(mistakes1)
        num_err_history.append(num_errs)
        
        print("Number of errors in iteration %d:%d\n"%(iterator,num_errs))
        print("weights\n",w)
        
def eval_perceptron(neg_examples,pos_examples,w):
    num_neg_examples =  tf.size(neg_examples[0])
    num_pos_examples =  tf.size(pos_examples[0])
    mistakes0 = []
    mistakes1 = []

    for i in range(num_neg_examples.eval()):
        x = neg_examples[i]
        activation = tf.multiply(x,w)
        if activation.eval() >= 0:
           mistakes0.append(i)     
    for i in range(num_pos_examples.eval()):
        x = pos_examples[i]
        activation = tf.multiply(x,w)
        if activation.eval() < 0:
            mistakes1.append(i)
    return mistakes0,mistakes1
            
def update_weights(neg_examples,pos_examples,w_current):
    num_neg_examples =  tf.size(neg_examples[0])
    num_pos_examples =  tf.size(pos_examples[0])
    w = w_current
    
    for i in range(num_neg_examples):
        x = neg_examples[i]
        activation = tf.multiply(x,w)
        if activation.eval() >= 0:
            w = w - x
            
    for i in range(num_pos_examples):
        x = pos_examples[i]
        activation = tf.multiply(x,w)
        if activation.eval() < 0:
            w = w + x
            

neg = tf.placeholder(tf.float32,[None,2],name="neg-input")
pos = tf.placeholder(tf.float32,[None,2],name="pos-input")
w_init = tf.placeholder(tf.float32,[3,1],name="w_init")
w_gen = tf.placeholder(tf.float32,[3,1],name="w_gen")
    
main = learn_perceptron(neg,pos,w_init,w_gen)  
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    filename=u'data/dataset1.mat'
    data=sio.loadmat(filename)
    print(data.neg_examples_nobias)
    # sess.run(main,feed_dict={neg:data.})