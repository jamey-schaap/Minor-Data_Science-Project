import pandas as pd
from configs.data import MACHINE_LEARNING_DATASET_PATH
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from configs.enums import Column, RiskClassifications

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


def split_data(dataframe: pd.DataFrame):
    data_by_risk = [dataframe[dataframe["country_risk"] == v] for v in RiskClassifications.get_values()]
    split_data = [
        # Train (60%), validation (20%) and test (20%) datasets
        np.split(sd.sample(frac=1, random_state=0), [int(0.6 * len(sd)), int(0.8 * len(sd))])
        for sd
        in data_by_risk
    ]

    train = pd.concat([row[0] for row in split_data])
    valid = pd.concat([row[1] for row in split_data])
    test = pd.concat([row[2] for row in split_data])

    return train, valid, test


def main():
    print("Loading dataset...")
    df = pd.read_excel(MACHINE_LEARNING_DATASET_PATH)

    train, valid, test = split_data(df)

    train, X_train, y_train = scale_dataset(train, oversample=True)
    valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
    test, X_test, y_test = scale_dataset(test, oversample=False)

    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    from sklearn.linear_model import LogisticRegression
    lg_model = LogisticRegression()
    lg_model.fit(X_train, y_train)
    y_pred = lg_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    from sklearn.svm import SVC
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    main()
