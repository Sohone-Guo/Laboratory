import math
import numpy as np
# Convoluation

class CNN:
    def __init__(self, size_filter=3*3, number_filter=10, stride=1, size_pooling = 3*3):
        
        # Setting the convolution
        self.size_of_filter = int(math.sqrt(size_filter))
        self.number_of_filter = number_filter
        self.stride = stride

        # Setting the pooling
        self.size_of_pooling = size_pooling

    def trainning(self,data):

        # Setting the image 
        input_, row, column =[], data.shape[0], data.shape[1]
        for i in range(0,row-self.size_of_filter,self.stride):
            for j in range(0,column-self.size_of_filter,self.stride):
                input_.append(data[i:i+self.size_of_filter,j:j+self.size_of_filter].reshape(-1).tolist())
        
        # Make a input of convoluation
        # Transfer the data to 3x3
        input = np.asarray(input_)

        # Initial the convoluation filter
        filter = np.random.random_sample((self.number_of_filter,self.size_of_filter**2))

        # then inner dot
        filter_after = input.dot(filter.T)

        # Initial pooling matrix
        pooling_input = []
        for item in range(filter_after.shape[1]):
            pooling_input_ = []
            tp_ = filter_after[:,item].reshape(-1,column-self.size_of_filter)
            tp_row, tp_column = tp_.shape
            
            # Extract the max value
            for i in range(0,tp_row-self.size_of_filter,self.stride):
                for j in range(0,tp_column-self.size_of_filter,self.stride):
                    pooling_input_.append(tp_[i:i+self.size_of_filter,j:j+self.size_of_filter].max())
            
            pooling_input.append(pooling_input_)

        pooling_input = np.asarray(pooling_input)
        print(pooling_input.shape)

        DNN_input = pooling_input.reshape(1,-1)
        # Begin the DNN
        Dnn = DNN(DNN_input.shape[1],1000,1)
        Dnn.cost_function_derivative(DNN_input,np.asarray([[1]]))

        # print(filter_after.shape)

class DNN:
    def __init__(self,number_input,first_layer,number_output):
        print("DNN")
        # Initial numbers
        self.number_input = number_input
        self.number_first_layer = first_layer
        self.number_output = number_output
        self.weight_1 = np.random.random_sample((number_input,first_layer))
        self.weight_2 = np.random.random_sample((first_layer,number_output))
        
    def forward_propagation(self,input):
        # forward propagation
        # initial the input as numpy array
        input = np.asarray(input)

        self.z2 = input.dot(self.weight_1)
        self.layer_1 = self.sigmoid(self.z2)
        self.z3 = self.layer_1.dot(self.weight_2)
        self.y_ = self.sigmoid(self.z3)
        return self.y_

    def sigmoid(self,z):

        # return the activity function
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self,z):

        # return derivative of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def cost_function_derivative(self,input,y):
        for i in range(10):
            self.y_ = self.forward_propagation(input)
            delta3 = -(y.T-self.y_)*self.sigmoid_derivative(self.z3)
            dJdw2 = np.dot(self.layer_1.T, delta3)

            delta2 = np.dot(delta3, self.weight_2.T)*self.sigmoid_derivative(self.z2)
            dJdw1 = np.dot(input.T,delta2)

            self.weight_1 -= dJdw1
            self.weight_2 -= dJdw2
            print("Error: ",sum(0.5*(y.T-self.y_)**2)[0])

        return self.forward_propagation(input)



    


        





