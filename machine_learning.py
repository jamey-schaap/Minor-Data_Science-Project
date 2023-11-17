import pandas as pd
from configs.data import MERGED_DATASET_PATH
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


# Feature vector
# Qualitative - categorical data (finite number of categories or groups)
#  - Nominal data
#    > no order
#    > one-hot encoding
#    > Example: Countries
#  - Ordinal data
#   > inherent order
#   > mark them numerically; ex: worse...better => 0...1000
#   > Example: Age bracket, happiness
# Quantitative - numerical data (discrete or continuous)

# Supervised learning
# 1. Classification - predict discrete classes
#   - Multiclass classification (multiple output labels); hotdog, pizza, icecream
#   - Binary classification (2 output labels): hotdog, NOT hotdog
# 2. Regression - predict continuous values
#   - Price of Ethereum, Temperature, Price of a house, etc...

# Preparing the data:
# 1. Fix default (null) values, Either:
#   - a: Predict those values based on other years
#   - b: Remove the rows with those values
# 2. Remove unused columns (features) or create a separate dataset containing only features (columns) used by the algorithm
# 3. one-hot encode columns: country, gov_type? (if used)
# 4. Add target (predicted category) column, where the risk factor is converted into low, high, ...
#   - Column in the main dataset as a string
#   - Numerical value (see Ordinal data) for each category in the machine learning/ algorithm dataset


def main():
    print("Loading dataset...")
    df = pd.read_excel(MERGED_DATASET_PATH)

    # Train (60%), validation (20%) and test (20%) datasets
    train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

    train, X_train, y_train = scale_dataset(train, oversample=True)
    valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
    test, X_test, y_test = scale_dataset(test, oversample=False)

    y_pred = ...

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
