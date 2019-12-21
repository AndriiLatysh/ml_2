import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import keras.models as keras_models
import keras.layers as keras_layers
import keras.utils as keras_utils
import keras.datasets as keras_datasets
import timeit


np.set_printoptions(linewidth=200)
(X_train, y_train), (X_test, y_test) = keras_datasets.mnist.load_data()

# image_index = 7777
# print(X_train[image_index])
# print(y_train[image_index])
# plt.imshow(X_train[image_index], cmap="Greys")
# plt.show()

X_train_count = X_train.shape[0]
X_test_count = X_test.shape[0]

print("Train size: {}".format(X_train_count))
print("Test size: {}".format(X_test_count))

X_train = X_train.reshape(X_train_count, 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(X_test_count, 28, 28, 1).astype("float32") / 255

y_train = keras_utils.to_categorical(y_train)
y_test = keras_utils.to_categorical(y_test)

CNN_model = keras_models.Sequential()

CNN_model.add(keras_layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
CNN_model.add(keras_layers.MaxPooling2D(pool_size=(2, 2)))

CNN_model.add(keras_layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
CNN_model.add(keras_layers.MaxPooling2D(pool_size=(2, 2)))

CNN_model.add(keras_layers.Flatten())

CNN_model.add(keras_layers.Dense(64, activation="relu"))
CNN_model.add(keras_layers.Dropout(0.2))

CNN_model.add(keras_layers.Dense(10, activation="softmax"))

CNN_model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=["accuracy"])

print(CNN_model.summary())

starting_time = timeit.default_timer()

CNN_model.fit(X_train, y_train, epochs=10, batch_size=100, shuffle=True)

ending_time = timeit.default_timer()

print("Training time: {} s".format(round(ending_time-starting_time)))

CNN_model_evaluation = CNN_model.evaluate(X_test, y_test, verbose=0)

print("Cross-validation accuracy: {}".format(CNN_model_evaluation[1]))

CNN_model.save("models/MNIST_CNN.h5")
