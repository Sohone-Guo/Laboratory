import Neural.neural
import numpy as np

# input 2 -> 3 -> 2
# input (2,3),(2,5),(0.5,0.7),(0.8,0.1) output (1,1,0,0)

input = np.asarray([[2,3],[2,5],[0.5,0.1],[0.6,0.7]])
y = np.asarray([[1,1,0,0]])

neural_model = Neural.neural.NeuralNetwork(2,3,1)
print(neural_model.cost_function_derivative(input,y))