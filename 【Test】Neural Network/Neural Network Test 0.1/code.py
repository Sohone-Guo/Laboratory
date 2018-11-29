
# Test the neural network

import NeuralNetwork.neuralnetwork
import numpy as np

# Read the dataset
data = np.loadtxt("data.csv",delimiter=',')

# Modify a sequence in-place by shuffling its contents
# Shuffle the datasets
_ = np.random.shuffle(data)

# split datasets to trainning and test
input = data[:-200,:-2]
output = data[:-200,-2:]

# test data
test_data = data[-200:,:-2]
y = data[-200:,-2:]


# DNN
# Initial the DNN
DNN = NeuralNetwork.neuralnetwork.DNN(4,10,2)
DNN.backward_propagation(input,100,100000000,output)

DNN.forward_propagatioin(test_data)
y_ = DNN.layer_output_[-1]
count = 0
for i in range(len(y)):
    # print(round(y_[i][1]))
    # print(y_[i].argmax(),y[i].argmax(),y_[i],y[i])
    if y_[i].argmax() == y[i].argmax() :
        count+=1

# print(len(y_),len(y))
print("The accuray is: ",count/len(y))
        