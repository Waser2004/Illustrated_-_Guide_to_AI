from .learner import Learner

class PerceptronLearner(Learner):
    def __init__(self, input_size, learning_rate):
        super().__init__(input_size, learning_rate)
    
    def optimize(self, inputs: list[float], label: float) -> float:
        prediction = self.predict(inputs)

        # correct prediction no action needed (Case 3 & Case 4)
        if (prediction > 0 and label > 0) or (prediction < 0 and label < 0):
            return prediction
        
        # handle Case 1: increase weights where input was high | decrease weights where input was low
        if prediction <= 0 and label > 0:
            for i in range(self.input_size):
                if inputs[i] > 0:
                    self.weights[i] += self.learning_rate
                else:
                    self.weights[i] -= self.learning_rate
            
            self.bias += self.learning_rate

            return prediction

        # handle Case 2: decrease weights where input was high | increase weights where input was low
        for i in range(self.input_size):
            if inputs[i] > 0:
                self.weights[i] -= self.learning_rate
            else:
                self.weights[i] += self.learning_rate
        
        self.bias -= self.learning_rate

        return prediction

        
    def predict(self, inputs: list[float]) -> float:
        sum = 0
        for x, w in zip(inputs, self.weights):
            sum += x * w

        return sum + self.bias