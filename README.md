# OCR Handwritten Digit Classification

This repository contains a small Python project for handwritten digit recognition using OpenCV and TensorFlow/Keras.

## Contents

- `main.py` — loads a digit grid image, splits it into individual digit cells, prepares training and test datasets, and trains a model.
- `digits1.png` — expected digit sample image used for grid extraction.
- `venv/` — project virtual environment.

## Requirements

- Python 3.10+ (recommended)
- OpenCV (`opencv-python`)
- NumPy
- TensorFlow / Keras

## Setup

1. Activate the virtual environment:

   ```powershell
   .\venv\Scripts\Activate
   ```

2. Install required packages if not already installed:

   ```powershell
   pip install opencv-python numpy tensorflow
   ```

## Usage

Run the main script from the project directory:

```powershell
python main.py
```

## Notes

- The script expects `digits1.png` to contain a grid of digit images that can be split evenly into `50` rows and `100` columns.
- Each resulting cell is treated as one digit sample.
- If the image does not have the correct dimensions, the script will raise a descriptive error.

## Troubleshooting

- `FileNotFoundError: digits1.png not found` — put the sample image next to `main.py` or update the `image_path` in the script.
- `ValueError: Image shape ... is not divisible by the expected grid` — use a properly sized digit grid image.
- `TensorFlow` model errors for tiny image cells — the script now falls back to a dense classifier if cells are too small for a Conv2D network.

## License

This project is provided as-is for learning and experimentation.
