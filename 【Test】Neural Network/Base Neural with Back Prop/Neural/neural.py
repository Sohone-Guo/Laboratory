import numpy as np

# Build a neural network
# Build a class


class NeuralNetwork():
    def __init__(self,n_input,n_layer,n_output):
        
        # Build W values for 2 layers
        # First W n_input x n_layer
        # Second W n_layer x n_output
        self.weight_1 = np.random.random_sample((n_input,n_layer))
        self.weight_2 = np.random.random_sample((n_layer,n_output))

    def forward_propagation(self,input):
        # forward propagation
        input = np.asarray(input)

        # first layer, input x first w
        # second layer, first layer x second w
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
        for i in range(10000):
            self.y_ = self.forward_propagation(input)
            delta3 = -(y.T-self.y_)*self.sigmoid_derivative(self.z3)
            dJdw2 = np.dot(self.layer_1.T,delta3)
            # print(y.T)
            delta2 = np.dot(delta3,self.weight_2.T)*self.sigmoid_derivative(self.z2)
            dJdw1 = np.dot(input.T,delta2)

            self.weight_1 -= dJdw1
            self.weight_2 -= dJdw2
            # print(sum(0.5*(y.T-self.y_)**2)[0])

        return self.forward_propagation(input)
    
    