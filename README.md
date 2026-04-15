# Count Number of Objects

This project demonstrates a simple object counting example using OpenCV and image processing techniques. The provided script loads an image, converts it to grayscale, applies smoothing and edge detection, and counts the detected contours to estimate how many objects are present.

## Features

- Reads an input image (`coins.jpg`)
- Converts the image to grayscale
- Applies Gaussian blur and Canny edge detection
- Finds and draws contours around detected objects
- Prints the number of objects found in the image

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install -r reqirements.txt
```

## Usage

Run the main script with:

```bash
python main.py
```

The script will display the processed grayscale image and print the detected object count to the console.

## Notes

- Ensure `coins.jpg` is present in the same folder as `main.py`
- The current implementation counts external contours and works best for images with clearly separated objects

## Project structure

- `main.py` — main object counting script
- `reqirements.txt` — dependency list
- `README.md` — project documentation