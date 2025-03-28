import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("mnist_digit_model.h5")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # increaing one dimension for the kernel operation
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# print(x_test[0])
predictions = model.predict([x_testr])
print(y_test[0])
print(np.argmax(predictions[0]))