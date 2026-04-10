import math
import numpy as np

class NaiveSimpleNeuralNetwork():
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.weights = [[0.0 for _ in range(self.num_inputs)] for _ in range(self.num_outputs)]
        self.biases  = [0.0 for _ in range(self.num_outputs)]

    def train(self, inputs: list[float], label: int, learning_rate: float):
        """Train the neural network using backpropagation and return the loss."""
        # create expected and actual output vectors
        expected_y = [1 if i == label else 0 for i in range(self.num_outputs)]
        actual_y   = self.predict(inputs)

        # calculate gradients
        weight_gradients = [[0.0 for _ in range(self.num_inputs)] for _ in range(self.num_outputs)]
        bias_gradients   = [0.0 for _ in range(self.num_outputs)]

        for i in range(self.num_outputs):
            for j in range(self.num_inputs):
                weight_gradients[i][j] = inputs[j] * (actual_y[i] - expected_y[i])
            bias_gradients[i] = (actual_y[i] - expected_y[i])

        # update parameters
        for i in range(self.num_outputs):
            for j in range(self.num_inputs):
                self.weights[i][j] += -weight_gradients[i][j] * learning_rate
            self.biases[i] += -bias_gradients[i] * learning_rate

        # return cross-entropy loss
        return -math.log(actual_y[label])      

    def predict(self, inputs: list[float]) -> list[float]:
        """Predict the output of the neural network given the inputs."""
        neuron_outputs = [0 for _ in range(self.num_outputs)]

        # apply weights
        for i, weights in enumerate(self.weights):
            for j, weight in enumerate(weights):
                neuron_outputs[i] = inputs[j] * weight

        # apply bias
        for i, bias in enumerate(self.biases):
            neuron_outputs[i] += bias

        # softmax
        outputs = []
        for i in range(self.num_outputs):
            outputs.append(self.softmax(i, neuron_outputs))

        return outputs

    @staticmethod
    def softmax(index: int, values: list[float]) -> float:
        """Calculate the softmax value for a given index and list of values."""
        assert 0 <= index < len(values), "Index out of bounds for softmax function."

        return math.e ** values[index] / sum([math.e ** value for value in values])


class MatrixSimpleNeuralNetwork():
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # Note that we initialize the weights differntly here, such that the weights for one neuron are stored in one row of the weights matrix. 
        # This allows us to use matrix multiplication to calculate the neuron outputs.
        self.weights = np.array([[0.0 for _ in range(self.num_outputs)] for _ in range(self.num_inputs)])
        self.biases  = np.array([0.0 for _ in range(self.num_outputs)])

    def train(self, inputs: list[float], label: int, learning_rate: float):
        """Train the neural network using backpropagation and return the loss."""
        # create expected and actual output vectors
        expected_y = np.array([1 if i == label else 0 for i in range(self.num_outputs)])
        actual_y   = np.array(self.predict(inputs))

        # calculate gradients
        weight_gradients = np.outer(np.array(inputs), (actual_y - expected_y))
        bias_gradients   = actual_y - expected_y

        # update parameters
        self.weights += -weight_gradients * learning_rate
        self.biases  += -bias_gradients * learning_rate

        # return cross-entropy loss
        return -math.log(actual_y[label])

    def predict(self, inputs: list[float]) -> list[float]:
        """Predict the output of the neural network given the inputs."""
        # apply weights and bias using matrix multiplication
        inputs_vector  = np.array(inputs)
        neuron_outputs = inputs_vector @ self.weights + self.biases

        # softmax
        exp_values = np.exp(neuron_outputs)
        return (exp_values / sum(exp_values)).tolist()