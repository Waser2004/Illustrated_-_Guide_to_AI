import sys
from typing import List, Tuple

# Select learner here: True = Perceptron, False = LMS
USE_PERCEPTRON = True

PERCEPTRON_LEARNING_RATE = 1
MSE_LEARNING_RATE        = 0.2


def parse_dataset(path: str) -> Tuple[int, int, List[Tuple[List[float], float]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    header = lines[0].split()
    n = int(header[0])
    m = int(header[1])

    data = []
    for line in lines[1:1+m]:
        parts = line.split()
        if len(parts) < n + 1:
            raise ValueError(f"expected {n+1} values per example, got: {line}")

        inputs = [float(x) for x in parts[:n]]
        label = float(parts[n])
        data.append((inputs, label))

    return n, m, data

def main():
    # load data
    path = "example.txt"
    
    try:
        n, m, data = parse_dataset(path)
    except Exception as e:
        print(f"Failed to parse dataset: {e}")
        sys.exit(1)

    # import learners after parsing so package path is correct
    if USE_PERCEPTRON:
        from learners.perceptron_learning import PerceptronLearner
        learner = PerceptronLearner(n, PERCEPTRON_LEARNING_RATE)
    else:
        from learners.LMS_learning import LMSLearner
        learner = LMSLearner(n, MSE_LEARNING_RATE)

    # train
    for sample in data:
        input, label = sample
        prediction = learner.optimize(input, label)

        is_correct = (prediction > 0 and label > 0) or (prediction < 0 and label < 0)
        print(f"Input: {input} | Predicted: {prediction} | Correct: {is_correct} | New weights: {learner.weights + [learner.bias]}")

    print("Final weights:", learner.weights + [learner.bias])

if __name__ == "__main__":
    main()
