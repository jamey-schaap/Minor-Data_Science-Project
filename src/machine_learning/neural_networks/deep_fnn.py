from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, History
from src.machine_learning.neural_networks.utils import get_last_layer_units_and_activation, plot_history
from src.machine_learning.utils import split_data, scale_dataset
import numpy as np
from sklearn.metrics import classification_report
import os
from typing import Optional, Callable
from src.machine_learning.neural_networks.learning_rate_schedulers import *
from src.configs.data import MODELS_PATH, VERSION


def deep_fnn_model(
        layers: int,
        units: int,
        dropout_rate: float,
        input_shape: int,
        num_classes: int) -> models.Sequential:
    """Creates an instance of a FNN model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        A FNN model instance.
    """
    op_units, op_activation = get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation="relu"))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def train_deep_fnn_model(dataframe,
                         learning_rate: float | Callable[[int], float] = 1e-3,
                         epsilon: float = 1e-07,
                         beta_1: float = 0.9,
                         beta_2: float = 0.999,
                         weight_decay: Optional[float] = None,
                         clipnorm: Optional[float] = None,
                         clipvalue: Optional[float] = None,
                         epochs: int = 1000,
                         batch_size: int = 128,
                         layers: int = 2,
                         units: int = 64,
                         dropout_rate: float = 0.2,
                         patience: int = 2,
                         verbose: int = 2,
                         file_name: Optional[str] = None,
                         disable_save: bool = False,
                         disable_plot_history: bool = False,
                         disable_print_report: bool = False) -> [models.Sequential, History, int]:
    """Trains FNN model on the given dataset.

    # Arguments
        dataframe: pandas dataframe containing train, test and validation data.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    train, valid, test = split_data(dataframe)

    train, x_train, train_labels = scale_dataset(train, oversample=True)
    valid, x_val, val_labels = scale_dataset(valid, oversample=False)
    test, x_test, test_labels = scale_dataset(test, oversample=False)

    # Verify that validation labels are in the same range as training labels.
    num_classes = len(np.unique(dataframe[dataframe.columns[-1]].values))
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError("Unexpected label values found in the validation set:"
                         " {unexpected_labels}. Please make sure that the "
                         "labels in the validation set are in the same range "
                         "as training labels.".format(
            unexpected_labels=unexpected_labels))

    # Create model instance.
    model = deep_fnn_model(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
        num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = "binary_crossentropy"
    else:
        loss = "sparse_categorical_crossentropy"

    if isinstance(learning_rate, float) or isinstance(learning_rate, int):
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay,
            epsilon=epsilon,
            clipnorm=clipnorm,
            clipvalue=clipvalue)

        # Create callback for early stopping on validation loss.
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience),
        ]

    else:
        optimizer = Adam(
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay,
            epsilon=epsilon,
            clipnorm=clipnorm,
            clipvalue=clipvalue)

        # Create callback for early stopping on validation loss.
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience),
            LearningRateScheduler(learning_rate)
        ]

    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    # Train and validate model.
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        verbose=verbose,
        batch_size=batch_size)

    # Print results.
    history = history.history
    if not disable_plot_history:
        print("Validation accuracy: {acc}, loss: {loss}".format(
            acc=history["val_acc"][-1], loss=history["val_loss"][-1]))

        plot_history(history, num_classes)

    if not disable_print_report:
        y_pred = model.predict(x_test).argmax(axis=1)
        print(classification_report(test_labels, y_pred))

    # Save model.
    if not disable_save:
        # tf-version_Optimizer_layers_units_dropout_learning-rate_epochs
        file_name = f"{VERSION}_Adam_{layers}_{units}_{dropout_rate}_{learning_rate}_{epochs}.deep_fnn.keras" if file_name is None else file_name
        model.save(os.path.join(MODELS_PATH, file_name))
        print(f"\nModel has been saved as '{file_name}'")

    return model, history, num_classes