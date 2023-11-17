import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import numpy as np


def scale_dataset(dataframe: pd.DataFrame, oversample: bool = False):
  # if target column is the last value
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
      ros = RandomOverSampler()
      X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))
  return data, X, y


def plot_loss(history):
  fig, ax = plt.subplots(nrows=1, ncols=1)
  ax.plot(history.history["loss"], label="loss")
  ax.plot(history.history["val_loss"], label="val_loss")
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Binary crossentropy")
  ax.legend()
  ax.grid(True)
  fig.savefig("./out/loss.png")
  plt.close()



def plot_accuracy(history):
  fig, ax = plt.subplots(nrows=1, ncols=1)
  ax.plot(history.history["accuracy"], label="accuracy")
  ax.plot(history.history["val_accuracy"], label="val_accuracy")
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Accuracy")
  ax.legend()
  ax.grid(True)
  fig.savefig("./out/accuracy.png")
  plt.close()


def main():
  print("Loading dataset...")
  df = pd.read_excel("./MachineLearning-Dataset-V1.xlsx")

  # Train (60%), validation (20%) and test (20%) datasets
  train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

  train, X_train, y_train = scale_dataset(train, oversample=True)
  valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
  test, X_test, y_test = scale_dataset(test, oversample=False)

  nn_model = tf.keras.Sequential([
   tf.keras.layers.Dense(32, activation="relu", input_shape=(9,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
  ])

  nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), 
    loss="binary_crossentropy", 
    metrics=["accuracy"])

  history = nn_model.fit(
    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2
  )
  # verbose=0

  plot_loss(history)
  plot_accuracy(history)



if __name__ == "__main__":
  main()
