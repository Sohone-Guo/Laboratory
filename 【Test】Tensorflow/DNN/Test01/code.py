
# This builded a DNN and trained a dataset
 
import tensorflow as tf
import numpy as np

#  Load the datasets
dataset = np.loadtxt("data_banknote_authentication.txt",delimiter=",")


# Using tensorflow build the DNN
# One layer DNN, and the linear model: y_=Wx+b

# Input
input = tf.placeholder(tf.float32,[None,4])
y = tf.placeholder(tf.float32,[None,2])
# First hidden layer
W_1 = tf.Variable(tf.random_uniform([4,10]))
b_1 = tf.Variable(tf.random_uniform([10]))

a_1 = tf.nn.relu(tf.matmul(input,W_1)+b_1)

# Second hidden layer
W_2 = tf.Variable(tf.random_uniform([10,2]))
b_2 = tf.Variable(tf.random_uniform([2]))

a_2 = tf.nn.relu(tf.matmul(a_1,W_2)+b_2)

# Loss function
cross_entropy = tf.reduce_sum(tf.square(a_2 - y)) #tf.reduce_mean(-tf.reduce_sum(y - a_2))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(a_2 - y, reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=a_2, logits=y))
# train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy)

# initial Variable
init = tf.global_variables_initializer()

# Session
sess = tf.Session()
sess.run(init)

# input and y
X = dataset[:,:-1]
output_ = dataset[:,-1]
output = np.zeros((len(output_),2))
output[np.arange(len(output_)),output_.astype("int")] = 1

# for i in range(10000000):
#     sess.run(train_step,{input:X,y:output})
#     if i % 100 == 0:
#         print("The epoch %d , error is: "%i,sess.run(cross_entropy,{input:X,y:output}))
epoch = 1000
batch_size = 10
for _ in range(epoch):
    for i in range(int(len(X)/batch_size)-1):
        batch_X = X[i*batch_size:(i+1)*batch_size,:]
        batch_y = output[i*batch_size:(i+1)*batch_size,:]

        _,coss = sess.run([train_step,cross_entropy],{input:batch_X,y:batch_y})
        print(coss)
sess.close()