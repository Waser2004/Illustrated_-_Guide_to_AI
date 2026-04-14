# Illustrated - Guide to AI (Implementations)

This repository contains code implementations and exercise solutions accompanying the book "Illustrated - Guide to AI" from THE WELCH LABS. The code is organised by chapter and aims to reproduce the examples and exercises in the book for learning and experimentation.

**Scope:** Implementations are educational and follow the book's material.

**Project structure (high-level)**

- [Chapter 1 - The Perceptron](Chapter%201%20-%20The%20Perceptron)
	- `main.py` - runnable example for the chapter.
	- `learners/` - learning algorithm implementations used by the chapter (perceptron learning and LMS/MSE learning).

- [Chapter 2 - Gradient Descent](Chapter%202%20-%20Gradient%20Descent)
	- `main.py` - samples parameter-space directions and plots a loss surface around a trained model.
	- `model.py` - simple CNN regressor definition.
	- `checkpoint.pt` - trained weights used for the loss-surface exploration.
	- `loss_surface.png` - generated contour visualization.

- [Chapter 3 - Backpropagation](Chapter%203%20-%20Backpropagation)
	- `README.md` - chapter overview and run instructions for both examples.
	- `linear model/main.py` - manual gradient updates for a tiny linear regression setup.
	- `city predictor/main.py` - training loop for a simple city classifier.
	- `city predictor/models.py` - naive and NumPy-based model implementations.
	- `city predictor/dataset.json` - tiny labeled dataset (Paris vs Berlin coordinates).

- [Chapter 4 - Deep Learning](Chapter%204%20-%20Deep%20Learning)
	- `README.md` - chapter overview, dependencies, and run instructions for the notebook.
	- `100_neuron_challenge.ipynb` - full challenge workflow: data extraction, architecture search, training, and decision-boundary visualization.
	- `checkpoint.pt` - trained model parameters used to reproduce the final visualization.
	- `data/` - source map image assets used to generate coordinate labels.

- [Chapter 5 - AlexNet](Chapter%206%20-%20AlexNet)
	- `feature_visualisation.py` - visualizes first-layer AlexNet filters and activations using the shared cube-regression checkpoint.
	- `filters.png` - generated visualization of first convolution filters.
	- `activations.png` - generated activation maps for an example image.

## Shared Model Origin (Chapter 2 and Chapter 5)

The model/checkpoint used by Chapter 2 and Chapter 5 is intentionally reused from the cube detection project in this repository (`Cube Detection model/`).

The original source repository is:

- https://github.com/Waser2004/Athena-Robot-AI

This model predicts how much of a cube is visible in an image (a regression target in the range [0, 1]).

These chapters focus on analyzing and visualizing behavior of the already trained model, rather than training a new model from scratch in this repository.
	