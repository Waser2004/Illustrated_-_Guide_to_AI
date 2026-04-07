class Learner:
    def __init__(self, input_size: int, learning_rate: float):
        self.input_size    = input_size
        self.learning_rate = learning_rate

        self.weights       = [0.0] * input_size
        self.bias          = 0


    def optimize(self, inputs: list[float], lable: float) -> float:
        """
            This function optimizes the model parameters based on the learning policy of the subclass.
            It returns predicted value before optimization.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, inputs: list[float]) -> float:
        """
            This function predicts the output based on the current model parameters.
        """
        raise NotImplementedError("Subclasses must implement this method")
        