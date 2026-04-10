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

## Note on Chapter 2 Model Origin

The Chapter 2 model is intentionally reused from a separate CNN project that predicts how much of a cube is visible in an image (a regression target in the range [0, 1]).

This chapter focuses on visualizing the optimization landscape (loss surface) for that already trained model, rather than training a new model from scratch in this repository.
	