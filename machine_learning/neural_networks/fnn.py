from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from machine_learning.neural_networks.utils import get_last_layer_units_and_activation, plot_history
from machine_learning.utils import split_data, scale_dataset
import numpy as np
from sklearn.metrics import classification_report
import os

def fnn_model(layers, units, dropout_rate, input_shape, num_classes):
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


def train_fnn_model(dataframe,
                    learning_rate=1e-3,
                    epochs=1000,
                    batch_size=128,
                    layers=2,
                    units=64,
                    dropout_rate=0.2,
                    patience=2,
                    verbose=2,
                    disable_save=False,
                    disable_plot_history=False,
                    disable_print_report=False):
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
    model = fnn_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = "binary_crossentropy"
    else:
        loss = "sparse_categorical_crossentropy"
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    # Create callback for early stopping on validation loss.
    callbacks = [EarlyStopping(
        monitor="val_loss", patience=patience)]

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
        model.save(os.path.join(os.environ["OUTPUT_PATH"], "fnn_model.keras"))

    return model, history["val_acc"][-1], history["val_loss"][-1]