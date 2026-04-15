import os
import cv2
import pytesseract
from matplotlib import pyplot as plt

image_path = os.path.join(os.path.dirname(__file__), "text_image.png")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Failed to load image from: {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# plt.figure(figsize=(10, 6))
# plt.imshow(image_rgb)
# plt.title("Original Image")
# plt.axis("off")
# plt.show()

possible_tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
for path in possible_tesseract_paths:
    if os.path.isfile(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

try:
    extracted_text = pytesseract.image_to_string(image_rgb)
    print(" Extracted Text:\n")
    print(extracted_text)
except pytesseract.pytesseract.TesseractNotFoundError:
    raise RuntimeError(
        "Tesseract OCR was not found. Install it from https://github.com/tesseract-ocr/tesseract "
        "and add it to your PATH, or update pytesseract.pytesseract.tesseract_cmd "
        "with the full path to tesseract.exe."
    )
