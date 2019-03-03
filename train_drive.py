# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")


import os
import cv2
import pickle
import random
import argparse
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from CNN_models import SmallerVGGNet, AlexNet, VGGNet
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import img_to_array

# Function for generating additional training data
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Adam Optimizer
from tensorflow.python.keras.optimizers import Adam, SGD

# Input a set of class labels
# Transform our class labels into one-hot encoded vectors
# Allow us to take an integer class label prediction from
# our Keras CNN and transform it back into a human-readable label.
from sklearn.preprocessing import LabelBinarizer

from tensorflow.python.keras import layers

# Import face detection function
# from faceDetection import faceDetect


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Construct the argument parse and parse the arguments for command line
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
ap.add_argument("-e", "--epoch", type=int, default=10,
                help="epoch times")
args = vars(ap.parse_args())


# Initial configuration
EPOCHS = args['epoch']      # Total num of epochs for training CNN
INIT_LR = 1e-3              # Initial Learning Rate (This is default value for Adam
BS = 32                      # Batch size for each epoch
# IMAGE_DIMS = (96, 96, 1)    # Spacial dimensions for input images
IMAGE_DIMS = (224, 224, 1)
COLOR = 0
# data = []                   # Hold the processed images
# labels = []                 # Hold the processed labels

# Grab the image paths and randomly shuffle them
print("[INFO] loading images...")
data = np.load("data.npy")
labels = np.load("labels.npy")


# Convert array into NumPy array
# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

# Binarize(二值化) the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)


# Create training and testing data
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.15, random_state=33)


# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


# Initialize the model and optimizer
print("[INFO] Compiling model...")
# PS: SmallerVGGNet only can deal with 96 x 96 x 3 images
# model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb.classes_))
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# AlexNet
# model = AlexNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb.classes_))
# opt = SGD(lr=1e-2, momentum=0.9, decay=5e-4)
# BS = 128

# VGG16
model = VGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
BS = 128
# For only two classes you should use binary cross-entropy as the loss.
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# Train the network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1
)


# Save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# Save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])