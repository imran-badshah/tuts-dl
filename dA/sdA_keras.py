# https://blog.keras.io/building-autoencoders-in-keras.html

import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train = utils.normalize(x_train, axis=1)
x_test = utils.normalize(x_test, axis=1)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape, x_test.shape)

encoding_dim = 32
flattend_input_dim = 784

input_img = layers.Input(shape=(flattend_input_dim, ))
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = layers.Dense(flattend_input_dim, activation='sigmoid')(encoded)

autoencoder = models.Model(input_img, decoded)

# Separate encoder
encoder = models.Model(input_img, encoded)
# Separate decoder
encoded_input = layers.Input(shape=(encoding_dim, ))
decoder_layer = autoencoder.layers[-1]
decoder = models.Model(encoded_input, decoder_layer(encoded_input))

# Training:
# Model configured to per-pixel binary crossentropy loss
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test))

# Visualise
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
n_display = 10
plt.figure(figsize=(20, 4))
for i in range(n_display):
    # Display original
    ax = plt.subplot(2, n_display, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed
    ax = plt.subplot(2, n_display, i + 1 + n_display)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()