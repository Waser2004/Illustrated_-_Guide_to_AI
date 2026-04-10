# Chapter 3 - Backpropagation

This chapter contains two small educational projects that demonstrate gradient-based learning and backpropagation concepts:

- A minimal linear model trained with manual parameter updates.
- A simple city classifier (Paris vs Berlin) trained with a softmax output layer and cross-entropy loss.

## Files and Subfolders

- `linear model/main.py`: Trains a 1D linear model (`y = wx + b`) on tiny synthetic data using either MSE or absolute loss style updates.
- `city predictor/main.py`: Loads a tiny coordinate dataset and trains a simple neural network classifier.
- `city predictor/models.py`: Contains two model implementations:
  - `NaiveSimpleNeuralNetwork`: list-based implementation of forward pass and parameter updates.
  - `MatrixSimpleNeuralNetwork`: NumPy-based implementation using vectorized operations.
- `city predictor/dataset.json`: Small dataset of latitude/longitude samples labeled as Paris or Berlin.

## What This Chapter Shows

- How gradients drive updates of weights and biases.
- The difference between a scalar linear model update loop and a vectorized neural-network update.
- Softmax outputs for classification and cross-entropy loss minimization.

## Run

Run each example from its own folder.

### Linear model

```bash
cd "linear model"
python main.py
```

### City predictor

```bash
cd "city predictor"
python main.py
```

By default, `city predictor/main.py` runs the matrix-based model. You can switch to the naive version by changing `model_type` in `main.py`.

## Dependencies

- Python 3.x
- NumPy (required for the matrix model in `city predictor/models.py`)

Install NumPy if needed:

```bash
pip install numpy
```