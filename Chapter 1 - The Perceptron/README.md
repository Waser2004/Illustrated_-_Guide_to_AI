**Project Overview**

- **Description:** Simple, educational implementation of two basic linear learners (Perceptron and LMS/MSE) used to illustrate weight updates and learning rules for Chapter 1 of the guide.

- **Purpose:** Train a linear model on small datasets and observe weight/bias updates and predictions.

**Files**

- **Main script:** [main.py](main.py): parses the dataset, selects a learner, runs training and prints per-sample updates and final weights.
- **Learners package:** [learners/](learners/): contains reusable learner classes and specific learning algorithms implemented in
  - [learners/learner.py](learners/learner.py)
  - [learners/perceptron_learning.py](learners/perceptron_learning.py)
  - [learners/LMS_learning.py](learners/LMS_learning.py)

**Input format**

The dataset file (default: `example.txt`) must use this format:

```
n m
x_1 x_2 ... x_n y
...
x_1 x_2 ... x_n y
```

- `n`: number of input features per example
- `m`: number of training examples
- Each example line contains `n` numeric input values followed by the label `y` (a number). The learners treat positive labels as positive class and negative labels as negative class.

Simple example (2 inputs, 3 examples):

```
2 3
1 0 1
0 1 -1
1 1 1
```

**How to run**

1. Make sure `example.txt` is present in the repository root and follows the format above.
2. Run the main script:

```
python main.py
```

3. Toggle which learner runs by editing the variables at the top of `main.py`:

- Set `USE_PERCEPTRON = True` to use the Perceptron learner.
- Set `USE_PERCEPTRON = False` to use the LMS/MSE learner.

You can also adjust learning rates via `PERCEPTRON_LEARNING_RATE` and `MSE_LEARNING_RATE` constants in `main.py`.

**What `main.py` does**

- Parses the dataset with `parse_dataset(path)` (returns `n, m, data`).
- Dynamically imports the chosen learner class from the `learners` package and constructs an instance with `n` and the selected learning rate.
- For each sample it calls `learner.optimize(inputs, label)`, prints the raw prediction and whether the sign of prediction matches the sign of the label, and shows the updated weights and bias.

**Learners package (overview)**

- `learners/learner.py`: defines the base `Learner` class with:
  - `weights` (list of floats) and `bias` (float)
  - `optimize(inputs, label)` abstract: should update weights and return prediction
  - `predict(inputs)` abstract: should return current linear output (dot(weights, inputs) + bias)

- `learners/perceptron_learning.py`: `PerceptronLearner` implements the perceptron update rule:
  - `predict` returns the linear sum + bias.
  - `optimize` checks sign(prediction) vs sign(label):
    - If sign matches, no update.
    - If a positive label was predicted non-positive, it increases weights for positive inputs and decreases for non-positive inputs; bias is increased.
    - If a negative label was predicted non-negative, it decreases weights for positive inputs and increases for non-positive inputs; bias is decreased.

- `learners/LMS_learning.py`: `LMSLearner` implements a simple MSE gradient update (LMS):
  - Computes `error = label - prediction`, gradients ~= `-2 * error * input_i`.
  - Updates weights and bias proportional to the learning rate.

**Output meaning**

- Each training step prints: the input vector, the raw predicted value, a boolean `Correct` that checks if the sign of prediction matches the sign of the label, and the new weights and bias.
- Final weights are printed after training completes.

Where to look next: `main.py` and the files in `learners/` contain the implementation details referenced above.
