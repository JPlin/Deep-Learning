# -*- coding: utf-8 -*-
#example1.5.1.py
import tensorflow as tf
import numpy as np


sess = tf.InteractiveSession()

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


with tf.name_scope('input'):
    #train_input 为输入的数值，train_target 为输入数值对应的标签
    train_input=tf.placeholder(tf.float32,shape=[None,2],name="x-input")
    train_target=tf.placeholder(tf.float32,shape=[None,1],name="y-input")

#W 为权值
#   truncated_normal() 产生正太分布的数据，参数指定了 维度 [2,2] 还有 标准差 0.1
#b 为偏移量
#   创建一个常量tensor，按照给出value来赋值
#   可以用shape来指定其形状。value可以是一个数，也可以是一个list。 

with tf.name_scope('layer1'):
    #下面为第一个隐藏层，有两个中间神经元
    W1=tf.Variable(tf.truncated_normal([2,2],stddev=0.1))
    b1=tf.Variable(tf.constant(0.1,shape=[2]))
    fc1=tf.nn.sigmoid(tf.matmul(train_input,W1)+b1)
    variable_summaries(fc1)

with tf.name_scope('layer2'):
    #下面为第二个隐藏层，只有一个输出神经元
    W2=tf.Variable(tf.truncated_normal([2,1],stddev=0.1))
    b2=tf.Variable(tf.constant(0.1,shape=[1]))
    fc2=tf.nn.sigmoid(tf.matmul(fc1,W2)+b2)
    variable_summaries(fc2)

#ce 为采用 平方误差函数 作为 损失函数后得到的结果值
#ts 为采用 Adam 作为梯度下降的算法，事实证明Adam 是一种比较好的下降算法
ce=tf.reduce_mean(tf.square(fc2-train_target))
tf.summary.scalar('loss', ce)
ts=tf.train.AdamOptimizer(1e-2).minimize(ce)

'''
上面已经建立了一个有隐藏层的模型
下面开始一个新的对话，并初始化tensorflow变量
然后进行训练
'''

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('F:\Deep Learning\Learning\XOR\log', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

#一个封装了产生数据的类，可以产生数据，并伴随着数据对应的结果值
class GenDataXOR():
    #产生数据的维度
    def __init__(self,shape):
        self.shape=shape
    #根据input 得出 结果值
    def func(self,dt):
        if(dt[0]+dt[1]<0.5):
            rt=[0]
        elif((dt[0]+dt[1])>1.5):
            rt=[0]
        else:
            rt=[1]
        return rt
    #随机产生输入数据
    def GenData(self):
        #self.data=np.random.random(self.shape)
        self.data = np.random.randint(0,2,size=(self.shape))
        return self.data
    #产生输入数据的正确结果值
    def GenVali(self):
        self.vali=np.array(list(map(self.func,self.data)))
        return self.vali

#genData 为产生训练数据的对象
#tsD    为产生测试数据的对象
genData=GenDataXOR([50,2])
tsD=GenDataXOR([500,2])

#tsData 为测试的输入数据，tsVali 为测试的标签
tsData=tsD.GenData()
tsVali=tsD.GenVali()

#开始训练，训练6000次
for i in range(6000):
    data=genData.GenData()
    vali=genData.GenVali()
    summary,_ = sess.run([merged,ts],feed_dict={train_input:data,train_target:vali})
    writer.add_summary(summary, i)
    #每训练1000次就用测试的数据进行模型的评估
    if(i%50==0):       
        print(sess.run(ce,feed_dict={train_input:tsData,train_target:tsVali}))
        
writer.close()
'''
训练完毕
后面根据训练时产生的数据，进行图形的绘制
'''   
reW1=np.array(sess.run(W1.value()))
reb1=np.array(sess.run(b1.value()))
reW2=np.array(sess.run(W2.value()))
reb2=np.array(sess.run(b2.value()))
#print(reW1)
#print(reb1)
#print(reW2)
#print(reb2)
def sigmoid(rt):
    return 1/(1+np.exp(-rt))
def GenZ(X,Y):
    Z=np.zeros(np.shape(X))
    for ity in range(len(X)):
        for itx in range(len(X[0])):
            l1=np.matmul([X[ity,itx],Y[ity,itx]],reW1)+reb1
            l1f=sigmoid(l1)
            l2=np.matmul(l1f,reW2)+reb2
            l2f=sigmoid(l2)
            Z[ity,itx]=l2f[0]
    return Z
        
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn-darkgrid')
x=np.linspace(0,1,100)
y=np.linspace(0,1,100)
X,Y=np.meshgrid(x,y)
Z=GenZ(X,Y)
fig=plt.figure(1)
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,rstride=8,cstride=8, alpha=0.3)
ax.contour(X,Y,Z,zdir='z',offset=0, cmap=plt.cm.coolwarm)
def fmap(mm):
    if(mm>0.5):
        return 'm'
    else:
        return 'r'
st1=np.transpose(tsData)
plt.figure(2)
plt.scatter(st1[0],st1[1],color=list(map(fmap,tsVali)))
plt.show()