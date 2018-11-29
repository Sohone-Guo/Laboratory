# This project is for practice of CNN
# Made by Sohone
# Anconda 3.6; tensorflow; keras(download datasets)
# -------------------------------------------------

# We will use a image datasets, it explained below.
'''
Dataset explainatioin(if we need change to other datasets):
    One image:
    5x5
    [
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0.0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]
'''

# Download this datasets
# mnist have the datasets, keras.utils is for one-hot data
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# We use the tensorflow for this CNN
import tensorflow as tf

# load image datasets (60000,28,28)
(X_train, y_train),(X_test,y_test) = mnist.load_data()

# reshape the datasets to (60000,1,28,28)
# and as type of float32
# convert to [0,1]
X_train = X_train.reshape((X_train.shape[0],28,28,1))
X_test = X_test.reshape((X_test.shape[0],28,28,1))
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert value to one-hot
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


# Make a function for W
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# Make a function for bias
def bias_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# Make a function for convolution 2D
# (strides = [batch,y,x,channel]);(padding where image add 0 out the boundary)
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding="SAME")

# Make a function for Pooling where we choose the Max pooling
# (ksize is -> area)
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# Initial the input:x and output:y
x = tf.placeholder(tf.float32, [None,28,28,1])
y = tf.placeholder(tf.float32,[None,10])

# First convolution and pooling
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
conv1 = tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
pooling1 = max_pool(conv1)

# Second convolution and pooling
W_conv2 = weight_variable([5,5,32,32])
b_conv2 = bias_variable([32])
conv2 = tf.nn.relu(conv2d(pooling1,W_conv2)+b_conv2)
pooling2 = max_pool(conv2)

# Flatten the final pooling
flatten = tf.reshape(pooling2,[-1,7*7*32])

# Build the DNN
# First layer
W_fc1 = weight_variable([7*7*32,1024])
b_fc1 = bias_variable([1024])
layer1 = tf.nn.relu(tf.matmul(flatten, W_fc1)+b_fc1)

# Second layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
output = tf.nn.relu(tf.matmul(layer1,W_fc2)+b_fc2)

# Set the cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

# Set the Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Check the accuracy
correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Begin the sessioin
# epoch is 2000
# and the batch size is 50
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(2000):
        batch = 50
        for batch_item in range(int(X_train.shape[0]/batch)-1):
            X = X_train[batch_item*batch:(batch_item+1)*batch]
            y_ = y_train[batch_item*batch:(batch_item+1)*batch]
            
            train_step.run(feed_dict={x:X,y:y_})
            
            # print the accuracy
            train_accuracy = accuracy.eval(feed_dict={x:X_test[-50:],y:y_test[-50:]})
            print(train_accuracy)
