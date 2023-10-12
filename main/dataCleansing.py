import pandas as pd 
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


"""
The files is used to explore, clean and transform the data
Features will also include creation of new features 
Image flipping, rotation, zooming, etc. will be done using Keras
"""


# Create an ImageDataGenerator with data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1.0/255,    # Normalize pixel values to the range [0, 1]
    shear_range=0.2,    # Random shear transformations
    zoom_range=0.2,     # Random zooming
    horizontal_flip=True,  # Random horizontal flipping
)

# Load and preprocess the dataset (in this example, assume you have a directory of images)
train_generator = datagen.flow_from_directory(
    'your_train_data_directory',
    target_size=(224, 224),   # Resize images to a consistent size
    batch_size=32,            # Batch size for training
    class_mode='categorical'  # The type of labels (categorical for classification)
)