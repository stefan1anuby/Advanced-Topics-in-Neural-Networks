import time

import torch
from tqdm import tqdm

from Dataset import Dataset
from Activations import Activations
from Layer import Layer

class NeuralNetwork:

    def __init__(self, layers):
        
        self.testing_set = None
        self.validation_set = None
        self.training_set = None
        self.layers = layers

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        print(f"The device used is: {device}")
        self.device = torch.device(device=device)

    def add_layer(self, input_size, output_size, activation_callback):
        self.layers.append(Layer(input_size, output_size, activation_callback))

    def _nn_feedforward(self, input_values):
        self.layers[0].feedforward(input_values)
        for i in range(1, len(self.layers)):
            self.layers[i].feedforward(self.layers[i - 1].activated_output)

    def _nn_backpropagate(self, expected_result):
        weights_update = [
            torch.zeros(layer.weights.shape, device=self.device) for layer in self.layers
        ]
        biases_update = [
            torch.zeros(layer.biases.shape, device=self.device) for layer in self.layers
        ]

        #compute delta(error) for the output layer
        delta = self.layers[-1].activated_output - expected_result.unsqueeze(-1)
        biases_update[-1] = delta
        weights_update[-1] = torch.matmul(delta, self.layers[-2].activated_output.T)

        #backpropagate through the remaining layers (excluding input layer)
        for l in range(len(self.layers) - 2, 0, -1):
            delta = torch.matmul(self.layers[l + 1].weights.t(), delta) * self.layers[l].activation_derivative(self.layers[l].output)
            biases_update[l] = delta
            weights_update[l] = torch.matmul(delta, self.layers[l - 1].activated_output.T)
        
        #backpropagate to the first layer
        delta = torch.matmul(self.layers[1].weights.t(), delta) * self.layers[0].activation_derivative(self.layers[0].output)
        biases_update[0] = delta
        weights_update[0] = torch.matmul(delta, self.layers[0].input.T)

        return weights_update, biases_update

    def train_mini_batch(self, data_set, max_iterations=10, batch_size=10, learning_rate=0.01):

        batch_count = len(data_set) // batch_size

        for it in range(max_iterations):

            for i in tqdm(range(batch_count), unit=" mini batches", desc=f"Epoch {it + 1} / {max_iterations}"):

                weights_adjustments = [
                    torch.zeros(layer.weights.shape, device=self.device) for layer in self.layers
                ]
                biases_adjustments = [
                    torch.zeros(layer.biases.shape, device=self.device) for layer in self.layers
                ]
                batch = data_set[i * batch_size: (i + 1) * batch_size]

                for input_values, target in batch:
                    expected_result = torch.zeros(self.layers[-1].output_size, dtype=torch.float, device=self.device)
                    expected_result[target] = 1

                    self._nn_feedforward(input_values)

                    weights_update, biases_update = self._nn_backpropagate(expected_result)
                    for params_idx in range(len(weights_adjustments)):
                        weights_adjustments[params_idx] += weights_update[params_idx]
                        biases_adjustments[params_idx] += biases_update[params_idx]

                for idx, layer in enumerate(self.layers):
                    layer.weights -= learning_rate * weights_adjustments[idx] / batch_size
                    layer.biases -= learning_rate * biases_adjustments[idx] / batch_size

    def predict(self, input_values):
        self._nn_feedforward(input_values.reshape(self.layers[0].input_size, 1))
        return self.layers[-1].activated_output

    def test_model(self, test_set):
        wrong_predictions = 0
        correct_predictions = 0
        total_loss = 0.0

        for input_values, correct_tag in test_set:
            predicted = self.predict(input_values)
            predicted_value = torch.argmax(predicted)
            if predicted_value == correct_tag:
                correct_predictions += 1
            else:
                wrong_predictions += 1

            expected_result = torch.zeros(self.layers[-1].output_size, dtype=torch.float)
            expected_result[correct_tag] = 1
            total_loss += torch.nn.functional.cross_entropy(predicted.t(), expected_result.unsqueeze(-1).t())

        print(f"Correct: {correct_predictions}, "
              f"Wrong: {wrong_predictions}, "
              f"Total: {correct_predictions + wrong_predictions}, "
              f"Accuracy: {int(correct_predictions / (correct_predictions + wrong_predictions) * 10000.) / 100}%, "
              f"Average Loss: {total_loss / len(test_set)}\n")

        time.sleep(1)


if __name__ == '__main__':
    dataset = Dataset("mnist.pkl.gz")
    
    model = NeuralNetwork([
        Layer(784, 100, Activations.get_activation_function("sigmoid")),
        Layer(100, 10, Activations.get_activation_function("softmax"))
    ])
    
    MAX_ITERATION = 1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.5
    model.train_mini_batch(dataset.training_set, MAX_ITERATION, BATCH_SIZE, LEARNING_RATE)

    print('Training set:')
    model.test_model(dataset.training_set)

    # Training set:
    # Correct: 49438, Wrong: 562, Total: 50000, Accuracy: 98.87 %, Average Loss: 1.4854742288589478

    print('Testing set:')
    model.test_model(dataset.testing_set)
    # Testing set:
    # Correct: 9745, Wrong: 255, Total: 10000, Accuracy: 97.45 %, Average Loss: 1.4963749647140503

    print('Validation set:')
    model.test_model(dataset.validation_set)
    # Validation set:
    # Correct:¬ 9751, Wrong: 249, Total: 10000, Accuracy: 97.51 %, Average Loss: 1.4948376417160034