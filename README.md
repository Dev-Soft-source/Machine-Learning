# Recognizing Handwritten Digits

This project demonstrates a simple handwritten digit recognition example using scikit-learn and the built-in `digits` dataset.

## What it does

- Loads the `digits` dataset from `sklearn.datasets`
- Flattens each image from 8x8 pixels into a 1D feature vector
- Trains an `MLPClassifier` neural network on the first 1000 samples
- Evaluates accuracy on the remaining test samples
- Prints the final accuracy score

## Requirements

- Python 3.8+
- scikit-learn
- matplotlib

## Install dependencies

```bash
pip install scikit-learn matplotlib
```

## Run the script

```bash
python main.py
```

## Notes

- The current implementation uses a small dataset split and a simple MLP configuration.
- The script prints the model accuracy after training.
- Optional plotting code is included but commented out.
