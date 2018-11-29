# Neural Network
import numpy as np

class DNN:
    def __init__(self, *node_of_layers):
        self.layers = list(node_of_layers)
        
        # save the W value, index as the position of w
        self.W = self.build_W()

    def build_W(self):

        # build w 
        W = []
        for item in range(1,len(self.layers)):
            W.append(np.random.random_sample((self.layers[item-1],self.layers[item])))
        
        # print(W)
        # return the W
        return W


    def forward_propagatioin(self,input):
        
        # Inital the layer input
        layer_input = input
        self.layer_output_ = []
        self.z_ = []
        for w in self.W:
            z = layer_input.dot(w)
            layer_output = self.sigmoid(z)
            self.layer_output_ .append(layer_output)
            self.z_.append(z)

            layer_input = layer_output


    def backward_propagation(self,input,chunk,epochs,output):
        for epoch in range(epochs):
            np.random.shuffle(input)
            for item in range(0,input.shape[0]-chunk,chunk):
                X = input[item:item+chunk,:]
                y = output[item:item+chunk,:]
                
                # forward propagation
                self.forward_propagatioin(X)
                delta_last = -(y - self.layer_output_[-1]) * self.sigmoid_derivative(self.z_[-1])
                dJdw_last = np.dot(self.layer_output_[-2].T,delta_last)
                self.W[-1] -= dJdw_last
                for w_ in range(2,len(self.W)):
                    back_index_w = len(self.W) - w_

                    delta = np.dot(delta_last,self.W[back_index_w+1].T)*self.sigmoid_derivative(self.z_[back_index_w])
                    dJdw = np.dot(self.layer_output_[back_index_w-1].T,delta)
                    # print(dJdw)
                    self.W[back_index_w] -= dJdw
                    delta_last = delta

            self.forward_propagatioin(input)
            # print(self.W[1])
            if epoch%100==0:
                print("The %d epoch, error is: "%epoch,sum(0.5*(output - self.layer_output_[-1])**2)[0])

        return self.layer_output_[-1]



    def sigmoid_derivative(self,z):

        # return derivative of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
        




    def sigmoid(self,z):

        # return sigmoid
        return 1/(1+np.exp(-z))




    





