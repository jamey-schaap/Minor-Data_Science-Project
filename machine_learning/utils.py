import pandas as pd
import numpy as np
from configs.enums import RISKCLASSIFICATIONS
from typing import Tuple, Any
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def split_data(dataframe: pd.DataFrame) -> Tuple[np.array, np.array, np.array]:
    data_by_risk = [dataframe[dataframe["country_risk"] == v] for v in RISKCLASSIFICATIONS.get_values()]
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


def scale_dataset(dataframe: pd.DataFrame, oversample: bool = False) -> Tuple[np.ndarray[Any, np.dtype], np.ndarray, np.ndarray]:
    # Assuming target column is the last column
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y
