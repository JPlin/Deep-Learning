# coding:utf-8
#线性回归：监督学习算法
#聚类算法：进行初步分析:k-means
import matplotlib.pyplot as plt
import pandas as pd
#一个可视化库seaborn
import seaborn as sns
import numpy as np
import tensorflow as tf

num_puntos = 2000
conjunto_puntos = []
for i in range(num_puntos):
	if np.random.random() > 0.5:
		conjunto_puntos.append([np.random.normal(0.0,0.9),np.random.normal(0.0,0.9)])
	else:
		conjunto_puntos.append([np.random.normal(3.0,0.5),np.random.normal(1.0,0.5)])


df = pd.DataFrame({"x":[v[0] for v in conjunto_puntos],
			"y":[v[1] for v in conjunto_puntos]})
sns.lmplot("x","y",data=df,fit_reg=False,size = 6)
plt.show();

#使用k-means 算法对上面的数据进行分组
vectors = tf.constant(conjunto_puntos)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

#在增加维度是为了让各个维度的意义相同
expanded_vectors = tf.expand_dims(vectors , 0)
expanded_centroides = tf.expand_dims(centroides,1)

#欧拉公式用来计算距离
#reduce_sum 是求和，可以减少一个维度
#argmin 返回某一个维度的最小索引值
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors,expanded_centroides)),2),0)

means = tf.concat(0,[tf.reduce_mean(tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments,c)),[1,1])),reduction_indices=[1])for c in range(k)])

update_centroides = tf.assign(centroides,means)

init_op = tf.initialize_all_variables()

sess = tf.session()
sess.run(init_op)

for step in range(100):
	_,centroid_values,assignments_values = sess.run([update_centroides,centroides,assignments])

#检查assignment_values tensor 的结果

data = {"x":[],"y":[],"cluster":[]}

for i in range(len(assignments_values)):
	data["x"].append(conjunto_puntos[i][0])
	data["y"].append(conjunto_puntos[i][1])
	data["cluster"].append(assignments_values[i])	

df = pd.DataFrame(data)
sns.lmplot("x","y",data=df,fit_reg=False,size = 6,hue = "cluster",legend = False)
plt.show()

