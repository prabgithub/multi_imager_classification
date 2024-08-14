from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
from joblib import dump
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple


def create_augmented_images_generator(X_attributes: np.array,
                                      X_images: np.array,
                                      Y: np.array,
                                      opt: Dict = {},
                                      only_images: bool = False,
                                      multiple_inputs: bool = False) -> Tuple:
    """    Generates augmented images on the fly, will also yield the
    corrosponding attributes of the original image

    Arguments:
        X_attributes {np.array} -- [description]
        X_images {np.array} -- [description]
        Y {np.array} -- [description]

    Keyword Arguments:
        opt {Dict} -- image augmentation options (default: {{}})
        only_images {bool} -- can optionally not yield attributes
         (default: {False})
        multiple_inputs {bool} -- [description] (default: {False})


    Yields:
        Iterator[Tuple] -- [description]
    """

    # load image augmentation parameters
    ROT_RANGE = opt.get('ROT_RANGE', 90)
    WIDTH_SHIFT_RANGE = opt.get('WIDTH_SHIFT_RANGE', 0.3)
    HEIGHT_SHIFT_RANGE = opt.get('HEIGHT_SHIFT_RANGE', 0.3)
    SHEAR_RANGE = opt.get('SHEAR_RANGE', 10)
    ZOOM_RANGE = opt.get('ZOOM_RANGE', 0.3)
    HOR_FLIP = opt.get('HOR_FLIP', True)
    VER_FLIP = opt.get('VER_FLIP', True)
    BATCH_SIZE = opt.get('BATCH_SIZE', 32)

    num_samples: int = X_images.shape[0]

    while True:
        # create image generator
        datagen = ImageDataGenerator(
            rotation_range=ROT_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            shear_range=SHEAR_RANGE,
            zoom_range=ZOOM_RANGE,
            horizontal_flip=HOR_FLIP,
            vertical_flip=VER_FLIP,
            fill_mode='nearest')
        # shuffled indices, don't want potential model to learn order of data
        idx: int = np.random.permutation(num_samples)

        # where the augmentation happens
        batches = datagen.flow(
            X_images[idx], Y[idx], batch_size=BATCH_SIZE, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            # TODO: Multiple inputs assumes three images / one attribute
            # Multiple inputs refers to multiple image inputs
            # This is used for training a collab model
            # This could be made to be more flexible
            if multiple_inputs:
                if only_images:
                    # returns 3 images and the labels
                    yield [batch[0], batch[0]], batch[1]
                else:
                    # returns 3 images, attributes and the labels
                    yield [batch[0],
                           batch[0],
                           X_attributes[idx[idx0:idx1]]], batch[1]
            # Used for training a single ConvNet
            # also for the light weight collab model
            else:
                if only_images:
                    # only one set of images and the labels
                    yield [batch[0]], batch[1]
                else:
                    # one set of images and the attributes
                    yield [batch[0], X_attributes[idx[idx0:idx1]]], batch[1]

            idx0 = idx1
            if idx1 >= num_samples:
                break


def load_trained_model(path_to_model: path, trainable: bool = False) -> Model:
    """ Loads a snapshot of a model and removes the output layer.
    Will optionally freeze the model's weights.

    Arguments:
        path_to_model {path} -- file path to a previously trained model

    Keyword Arguments:
        trainable {bool} -- Should the weights be frozen (default: {False})

    Returns:
        {Model} -- TensorFlow (Keras) model with the output removed
    """
    from tensorflow.keras import Model
    from tensorflow.keras.models import load_model

    model: Model = load_model(path_to_model)
    model = Model(model.input, model.layers[-2].output)
    for l in model.layers:
        l.trainable = trainable
    return model


def standardize(Xtrain: np.array,
                Xval: np.array,
                Xtest: np.array,
               path_to_store) -> Tuple[np.array, np.array, np.array]:
    """Standardise features (mean of 0 and std of 1)

    Arguments:
        Xtrain {np.array} -- [description]
        Xval {np.array} -- [description]
        Xtest {np.array} -- [description]

    Returns:
        Tuple[np.array, np.array, np.array] -- [description]
    """

    rescaler = StandardScaler()

    Xtrain = rescaler.fit_transform(Xtrain.astype(np.float))
    Xval = rescaler.transform(Xval.astype(np.float))
    Xtest = rescaler.transform(Xtest.astype(np.float))

    # save for future use on new data
    dump(rescaler, path_to_store, compress=True)

    return Xtrain, Xval, Xtest


