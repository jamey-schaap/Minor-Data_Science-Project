import matplotlib.pyplot as plt
from typing import Tuple
import tensorflow as tf


def plot_history(history: tf.keras.history, num_classes: int) -> None:
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
