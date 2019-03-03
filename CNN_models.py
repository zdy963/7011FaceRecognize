# import the necessary packages
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras import backend as K


def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - 0.5 * (K.square(y_pred[:, 1, 0]) + K.square(y_pred[:, 2, 0])) + margin))


def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


class simpleNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # First convolutional layer with ReLU-activation and max-pooling.
        # kernel_size means size of filter, eg: the filter is 5x5
        # filters means the number of filter, in this case we have 16 filter
        # so the output depth equals to 16
        model.add(Conv2D(kernel_size=3, strides=1, filters=16, padding='same',
                         activation='relu', name='layer_conv1', input_shape=inputShape))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        # model.add(Dropout(0.25))

        # Second convolutional layer with ReLU-activation and max-pooling.
        model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                         activation='relu', name='layer_conv2'))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        # model.add(Dropout(0.25))

        # Third convolutional layer with ReLU-activation and max-pooling.
        model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                         activation='relu', name='layer_conv3'))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        # model.add(Dropout(0.25))

        # Fourth convolutional layer with ReLU-activation and max-pooling.
        model.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                         activation='relu', name='layer_conv4'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(0.25))

        # Flatten the 4-rank output of the convolutional layers
        # to 2-rank that can be input to a fully-connected / dense layer.
        model.add(Flatten())

        # First fully-connected / dense layer with ReLU-activation.
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Last fully-connected / dense layer with softmax-activation
        # for use in classification.
        model.add(Dense(classes, activation='softmax'))

        return model


class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",    # Conv layer has 32 filters, with 3x3 kernel
                         input_shape=inputShape))
        model.add(Activation("relu"))                   # Use RELU as activation function
        model.add(BatchNormalization(axis=chanDim))     # Normalize data
        model.add(MaxPooling2D(pool_size=(3, 3)))       # Use a size 3x3 pool to reduce dim from 96x96 to 32x32
        model.add(Dropout(0.35))                        # Dropout randomly disconnect nodes from curr to next layer

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))   # Increase filter size from 32 to 64
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))       # Reduce the pool size to 2x2
        model.add(Dropout(0.35))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.35))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


class VGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # FC => RELU
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


class AlexNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL
        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="valid", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # CONV => RELU => POOL
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # CONV => RELU => POOL
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model