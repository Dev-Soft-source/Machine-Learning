from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    model, X_test, y_test, predictions = train_model()
    evaluate_model(y_test, predictions)