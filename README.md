# Cartooning Image

This project demonstrates a simple image cartoonification pipeline using OpenCV. The script reads an input image, applies edge detection and smoothing, and then combines the results to produce a cartoon-style output.

## Features

- Load an image file using OpenCV
- Convert the image to grayscale and apply median blur
- Detect edges using adaptive thresholding
- Smooth the original colors with a bilateral filter
- Combine edges and color for a cartoon effect
- Display and save the cartoonized image

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)

## Setup

1. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python
   ```

## Usage

1. Place the input image in the project directory and name it `Screenshot.png`.
2. Run the script:
   ```bash
   python main.py
   ```
3. The cartoon result will appear in a window and will be saved as `cartoon_output.jpg`.

## Notes

- Update the input filename in `main.py` if you want to use a different image.
- Close the display window or press any key to exit the program.

## File

- `main.py` - Main cartoonification script using OpenCV
