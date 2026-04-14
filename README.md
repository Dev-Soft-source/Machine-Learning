# Handwritten Digit Classification

This repository contains a TensorFlow-based example for classifying handwritten digits from the MNIST dataset.

## Project Overview

- Loads the MNIST dataset using `tf.keras.datasets.mnist`
- Preprocesses the data by flattening 28x28 images into 784-dimensional vectors
- Normalizes pixel values to the range `[0, 1]`
- Trains a simple `tf.keras.Sequential` model with a single dense softmax output layer
- Evaluates accuracy on the test set
- Saves and reloads the trained model
- Predicts and displays a sample digit image

## Prerequisites

- Python 3.8+ recommended
- TensorFlow 2.x
- NumPy
- Matplotlib

## Install

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install directly:

```bash
pip install tensorflow numpy matplotlib
```

## Run

```bash
python main.py
```

## Notes

- The model expects flattened input vectors of shape `(None, 784)`.
- The script saves the trained model to `epic_num_reader.h5`.
- The example includes a visualization of one test digit using `matplotlib`.

## Files

- `main.py` - Main training, evaluation, and prediction script
- `README.md` - Project description and usage instructions
