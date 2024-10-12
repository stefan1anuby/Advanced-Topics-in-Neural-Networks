import torch
import torch.nn.functional as F

class Activations:
    @staticmethod
    def sigmoid(tensor):
        return 1.0 / (1.0 + torch.exp(-tensor))

    @staticmethod
    def sigmoid_derivative(tensor):
        return Activations.sigmoid(tensor) * (1 - Activations.sigmoid(tensor))

    @staticmethod
    def softmax(tensor):
        aux = torch.exp(tensor)
        return aux / aux.sum()
    
    @staticmethod
    def relu(tensor):
        return torch.maximum(torch.tensor(0.0), tensor)

    @staticmethod
    def relu_derivative(tensor):
        return (tensor > 0).float()

    @staticmethod
    def tanh(tensor):
        return torch.tanh(tensor)

    @staticmethod
    def tanh_derivative(tensor):
        return 1 - torch.tanh(tensor) ** 2
    
    @staticmethod
    def get_activation_function(name):
        """Returns the activation function and its derivative by name."""
        if name == "sigmoid":
            return Activations.sigmoid, Activations.sigmoid_derivative
        elif name == "softmax":
            return Activations.softmax, None
        else:
            raise ValueError(f"Unknown activation function: {name}")