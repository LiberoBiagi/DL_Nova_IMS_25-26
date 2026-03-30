import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_and_normalize(path, label):

    """ Helper function to load an image from disk, decode it, and normalize pixel values to the range [0, 1]. """

    # read image file from disk
    img = tf.io.read_file(path)

    # decode image into RGB
    img = tf.image.decode_image(img, channels=3, expand_animations=False) 

    # convert pixel values from 0-255 to 0-1
    img = tf.cast(img, tf.float32) / 255.0

    return img, label


def preprocess_v1 (train_df, val_df, test_df, BATCH_SIZE=64):

    """Preprocessing version 1:
    - Load images from file paths and convert author names to numeric labels
    - Normalize pixel values to [0, 1]
    - Randomly augment training images with horizontal flips and small rotations
    - Create batches of images for more efficient training
    
    It takes the split dataframes as input and returns TensorFlow datasets ready for training, validation, and testing."""
    
    # get all unique author names from the training set
    authors = sorted(train_df["author"].unique())

    # create a dictionary: author name -> number
    author_to_idx = {author: i for i, author in enumerate(authors)}

    # create a new numeric label column
    train_df["label"] = train_df["author"].map(author_to_idx)
    val_df["label"] = val_df["author"].map(author_to_idx)
    test_df["label"] = test_df["author"].map(author_to_idx)

    # create TensorFlow datasets from file paths and numeric labels
    train_ds = tf.data.Dataset.from_tensor_slices((train_df["image_path"].values,train_df["label"].values))
    val_ds = tf.data.Dataset.from_tensor_slices((val_df["image_path"].values,val_df["label"].values))
    test_ds = tf.data.Dataset.from_tensor_slices((test_df["image_path"].values,test_df["label"].values))

    # apply the load_and_normalize function to each (path, label) pair in the datasets
    train_ds = train_ds.map(load_and_normalize)
    val_ds = val_ds.map(load_and_normalize)
    test_ds = test_ds.map(load_and_normalize)

    # define a data augmentation pipeline
    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(0.03),])

    # apply augmentation only to the training dataset
    train_ds = train_ds.map(lambda img, label: (data_augmentation(img, training=True), label))
    

    # creating batches of images for more efficient training
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds


