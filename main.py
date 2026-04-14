import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model("mnist_997_model.h5")

def preprocess(img):
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Example usage:
img = cv2.imread("digit0.png", cv2.IMREAD_GRAYSCALE)

prediction = model.predict(preprocess(img))
print("Predicted digit:", np.argmax(prediction))