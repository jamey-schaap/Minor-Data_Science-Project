import matplotlib.pyplot as plt
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.configs.enums import Column, RISKCLASSIFICATIONS


def plot_history(history: tf.keras.callbacks.History, num_classes: int) -> None:
    """
    Plots the history of a given tensorflow.keras model.
    :param history: tf.keras.history, The history of a tensorflow.keras model.
    :param num_classes: int, The number of classes/labels.
    """

    # Loss graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["loss"], label="loss")
    ax1.plot(history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch")
    if num_classes == 2:
        ax1.set_ylabel("Binary crossentropy")
    else:
        ax1.set_ylabel("Sparse categorical crossentropy")
    ax1.grid(True)
    ax1.legend()

    # Accuracy graph
    ax2.plot(history["acc"], label="accuracy")
    ax2.plot(history["val_acc"], label="val_accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.legend()

    plt.show()


def get_last_layer_units_and_activation(num_classes: int) -> Tuple[int, str]:
    """
    Gets the # of units and the activation function for the last network layer.
    :param num_classes: int, The number of classes/labels.
    :return: Tuple[int, str], A tuple containing the units and the activation function.
    """
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    return units, activation


def get_tensorflow_version() -> str:
    """
    Gets the currently used tensorflow version.
    :return: str, The tensorflow version.
    """
    return tf.__version__


def split_data(dataframe: pd.DataFrame) -> Tuple[np.array, np.array, np.array]:
    """
    Splits the data into a train, validation and test dataset, where each label/class is spread equally over each
    dataset.
    :param dataframe: pandas.Dataframe, The dataframe to split.
    :return: Tuple[np.array, np.array, np.array], A tuple containing the train, validation and test dataset
    respectively.
    """
    data_by_risk = [dataframe[dataframe[Column.COUNTRY_RISK] == v] for v in RISKCLASSIFICATIONS.get_values()]
    equally_divided_data = [
        # Train (60%), validation (20%) and test (20%) datasets
        np.split(sd.sample(frac=1, random_state=0), [int(0.6 * len(sd)), int(0.8 * len(sd))])
        for sd
        in data_by_risk
    ]

    train = pd.concat([row[0] for row in equally_divided_data])
    valid = pd.concat([row[1] for row in equally_divided_data])
    test = pd.concat([row[2] for row in equally_divided_data])

    return train, valid, test
