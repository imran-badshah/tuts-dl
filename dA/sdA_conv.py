from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist


def prepareForModel(input):
    """Normalizes and resizes `input` retaining number of inputs
    
    resizes = n_inputs x `input.shape[1]` x `input.shape[2]` x 1
    """

    input = utils.normalize(input, axis=1)
    input = input.reshape((len(input), input.shape[1], input.shape[2], 1))
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


def visualise_encoded_subplot(input_images, n_cols, figsize=(20, 4)):
    """Visualise given encoded layer
    
    Arguments:
        input_images: encoded image. Can have N images
        n_cols: Number of columns (images from dataset `input_images` to display) in subplot
        figsize: tuple for scale of images
    """
    n_rows = 1
    image_cols = input_images.shape[1]
    image_rows = input_images.shape[2] * input_images.shape[3]
    plt.figure(figsize=figsize)
    for i in range(n_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(input_images[i].reshape(image_cols, image_rows).T) # TO-DO: Determine `input_shape`
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def load_data():
    "Loads train and test data"
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = prepareForModel(x_train)
    x_test = prepareForModel(x_test)
    print(x_train.shape, x_test.shape)
    return x_train, x_test


def main():
    x_train, x_test = load_data()
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    # x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    print(autoencoder.summary())

    # Tensor Board
    # `tensorboard --logdir=/tmp/sdA_conv`
    autoencoder.fit(
        x_train, x_train,
        epochs=50, batch_size=128,
        shuffle=True,
        callbacks=[TensorBoard(log_dir='/tmp/sdA_conv')],
    )

    # Visualise
    decoded_imgs = autoencoder.predict(x_test)
    visualise_subplot(
        n_rows=2, n_cols=10,
        input_images=x_test, output_images=decoded_imgs,
        input_shape=(28, 28), output_shape=(28, 28)
    )
    encoder = Model(input_img, encoded)
    encoded_imgs = encoder.predict(x_test)
    visualise_encoded_subplot(
        input_images=encoded_imgs,
        n_cols=10
    )


if __name__ == "__main__":
    main()