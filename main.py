# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset using TensorFlow 2.x API
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten and normalize the data
data = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
labels = y_train.astype(np.int32)
test_data = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0
test_labels = y_test.astype(np.int32)

max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]
print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)
# displaying dataset using Matplotlib
def display(i):
    img = test_data[i]
    plt.title('label : {}'.format(test_labels[i]))
    plt.imshow(img.reshape((28, 28)))
    plt.show()
# Create a simple linear classifier using tf.keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,))
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(data, labels, epochs=10, batch_size=100)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")

# Make prediction
prediction = np.argmax(model.predict(np.array([test_data[0]])), axis=1)[0]
print("prediction : {}, label : {}".format(prediction, test_labels[0]))

# if prediction == test_labels[0]:
#     display(0)

# Save the model   
model.save('epic_num_reader.h5')

# Load the model and make a prediction
new_model = tf.keras.models.load_model('epic_num_reader.h5')

# Flatten and normalize the original x_test images before prediction
x_test_flat = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0
predictions = new_model.predict(x_test_flat)

print('label -> ', y_test[2])
print('prediction -> ', np.argmax(predictions[2]))

plt.title('label: {}  pred: {}'.format(y_test[2], np.argmax(predictions[2])))
plt.imshow(x_test[2], cmap='gray')
plt.axis('off')
plt.show()