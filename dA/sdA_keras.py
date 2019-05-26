# https://blog.keras.io/building-autoencoders-in-keras.html

import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def prepareForModel(input):
    "Normalizes and Flattens `input` retaining number of inputs"

    input = utils.normalize(input, axis=1)
    input = input.reshape((len(input), np.prod(input.shape[1:])))
    return input


def visualise_subplot(n_rows, n_cols, input_images, output_images, input_shape, output_shape, figsize=(20, 4)):
    """Visualise given images
    
    Arguments:
        n_rows: Number of rows in subplot 
        n_cols: Number of columns (images) in subplot
        input_images: input image to compare. Can have N images
        output_images: output image to access. Can have N images
        input_shape: original size of input image
        output_shape:  original size of output image
        figsize: tuple for scale of images
    """
    n_display = n_cols
    plt.figure(figsize=figsize)
    for i in range(n_display):
        # Display original
        ax = plt.subplot(n_rows, n_display, i + 1)
        plt.imshow(input_images[i].reshape(input_shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed
        ax = plt.subplot(n_rows, n_display, i + 1 + n_display)
        plt.imshow(output_images[i].reshape(output_shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

(x_train, _), (x_test, _) = mnist.load_data()
x_train = prepareForModel(x_train)
x_test = prepareForModel(x_test)

print(x_train.shape, x_test.shape)

encoding_dim = 32
flattend_input_dim = np.prod(x_train.shape[1:])

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
visualise_subplot(
    n_rows=2, n_cols=10,
    input_images=x_test, output_images=decoded_imgs,
    input_shape=(28, 28), output_shape=(28,28))

