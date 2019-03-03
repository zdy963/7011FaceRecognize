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
from CNN_models import SmallerVGGNet
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import img_to_array

# Function for generating additional training data
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Adam Optimizer
from tensorflow.python.keras.optimizers import Adam

# Input a set of class labels
# Transform our class labels into one-hot encoded vectors
# Allow us to take an integer class label prediction from
# our Keras CNN and transform it back into a human-readable label.
from sklearn.preprocessing import LabelBinarizer