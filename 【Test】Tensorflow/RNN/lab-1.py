import tensorflow as tf
import numpy as np

# One hot encoding
# happy
h = [1,0,0,0]
a = [0,1,0,0]
p = [0,0,1,0]
y = [0,0,0,1]

# x_data.shape = (1,5,4)
x_data = np.array([[h,a,p,p,y]],dtype=np.float32)

# Setting the hidden size of cell
x = tf.placeholder(tf.float32,[None,5,4])
hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

outputs, states = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)

# Running
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(outputs,{x:x_data}))

'''
[[[-0.03160123 -0.0784775 ]
  [-0.13658462  0.0056797 ]
  [-0.03980277  0.05107106]
  [ 0.02528752  0.08239696]
  [ 0.09112277  0.07920606]]]
'''

print(sess.run(states,{x:x_data}))

'''
LSTMStateTuple(c=array([[ 0.19109565,  0.17243849]], dtype=float32), 
h=array([[ 0.09112277,  0.07920606]], dtype=float32))
'''
