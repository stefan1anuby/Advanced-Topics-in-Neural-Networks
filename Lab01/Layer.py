import math
import torch

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input = None
        self.output = None
        self.activated_output = None
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function, self.activation_derivative = activation[0], activation[1]
        
        self.weights = torch.randn(output_size, input_size) * math.sqrt(1 / input_size)
        self.biases = torch.randn(output_size, 1)

    def feedforward(self, input_values):
        self.input = input_values
        self.output = torch.matmul(self.weights, input_values) + self.biases
        self.activated_output = self.activation_function(self.output)