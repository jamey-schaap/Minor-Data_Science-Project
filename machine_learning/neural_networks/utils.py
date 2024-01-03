import matplotlib.pyplot as plt
from typing import Tuple

def plot_history(history, num_classes: int) -> None:
    """
     # Arguments
        history: tf.History, the history of a Tensorflow.Keras model.
        num_classes: int, number of output classes.
    """
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

    ax2.plot(history["acc"], label="accuracy")
    ax2.plot(history["val_acc"], label="val_accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.legend()

    plt.show()

def get_last_layer_units_and_activation(num_classes: int) -> Tuple[int, str]:
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    return units, activation

def get_tensorflow_version() -> str:
    import tensorflow as tf
    return tf.__version__