import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist

# divide the training and test data set 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()
# print(x_train[0])

# Normalize the data since images are in gray level ( 1 channel - 0 to  255) not coloured (RGB)
# preprocessing => x_train/255

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()
print(y_train[0])

#resize the image for convolutional operation
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # increaing one dimension for the kernel operation
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(x_trainr.shape)


#create the deep learning model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#create the neuaral network model

model = Sequential()

#First convolutional layer 0 1 2 3  (60000, 28, 28, 1)
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:])) # 64 is the number of filters, 3,3 is the size of the filter
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))  # 2,2 is the size of the filter

#Second convolutional layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

#Third convolutional layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

#Fully connected layer #1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

#Fully connected layer #2
model.add(Dense(32))
model.add(Activation("relu"))   # 10 is the number of classes

#Last fully connected layer
model.add(Dense(10))
model.add(Activation("softmax"))


# print(model.summary())
print("Total Training Samples: ", len(x_trainr))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(x_trainr, y_train, epochs = 50, validation_split = 0.3)

test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_acc)

print([x_testr])
predictions = model.predict([x_testr])
print(predictions)
print(np.argmax(predictions[0]))

model.save("mnist_digit_model.h5")

