from .learner import Learner

class LMSLearner(Learner):
    def __init__(self, input_size, learning_rate):
        super().__init__(input_size, learning_rate)
    
    def optimize(self, inputs: list[float], label: float) -> float:
        prediction = self.predict(inputs)
        error = label - prediction

        # calculate gradients
        gradients = []
        for i in range(self.input_size):
            gradients.append(-2 * error * inputs[i])

        # optimize weights
        for i in range(self.input_size):
            self.weights[i] += - gradients[i] * (self.learning_rate / 2)

        # optimze gradient
        self.bias += 2 * error * (self.learning_rate / 2)

        return prediction

    def predict(self, inputs: list[float]) -> float:
        sum = 0
        for x, w in zip(inputs, self.weights):
            sum += x * w

        return sum + self.bias