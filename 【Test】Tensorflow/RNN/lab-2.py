import tensorflow as tf
import numpy as np
from string import punctuation,octdigits

# the dataset is 
data = "Wikispeedia is an implementation of the Wiki Game with the additional research purpose of using the gathered data in an artificial intelligence program that automatically learns commonsense knowledge.The Wikipedia Game is a version of the game where the player has 2 minutes 30 seconds to get from one website to another. It then averages the number of clicks and time it takes to get to finishing page. The player with the best averages wins. The online player with the most wins is shown as the leader. Although it has a log-in feature, it is not necessary to log-in. It will gives a player without a username a random name when coming to the website. It is an implementation by Alex Clemesha which offers variations called Speedrace, Least Clicks, 6 degrees, Find Jesus in 5, and No using 'United States'.Wikipedia Maze. Wikipedia Maze was a version of the game that awards points and badges for both creating and solving puzzles. Every time a user solves a puzzle they are awarded points based on the average amount of clicks it takes to solve the puzzle. The harder the puzzle is the more points that are awarded. Each puzzle can be voted up or down by other players, based on whether or not they like the it, which awards points to the creator. Players are also awarded badges when reaching certain milestones."

# Generate the word index
word_index = {}
for number, each in enumerate(set(data.strip(octdigits).split())):
    word_index[each] = number

word_index["unknown"] = number + 1

# splite the word to list
word = []
for item in data.strip(octdigits).split():
    word.append(word_index[item])

# setting the input x and target y
x_data = word[:-1]
y_data = word[1:]
y_data = np.array(y_data).reshape(1,228)

# make the one hot vector
one_hot_x = np.eye(len(word_index))[x_data]

# and reshape (1,228,140)
one_hot_x = one_hot_x.reshape(1,-1,len(word_index))

# Setting the tensorflow X, Y
# RNN cell 
X = tf.placeholder(tf.float32,[None,228,140])
Y = tf.placeholder(tf.int32,[None,228]) 

cell = tf.contrib.rnn.BasicLSTMCell(num_units=228, state_is_tuple=True)

_init_state = cell.zero_state(1,dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(cell, X, initial_state=_init_state,dtype=tf.float32)

# set weight
weight = tf.ones([1,228])

# declare the loss 
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y,weights=weight)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs,axis=2)

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l,_ = sess.run([loss,train],{X:one_hot_x, Y:y_data})
        print(i," loss:",l)