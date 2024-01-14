import pandas as pd
import numpy as np
from configs.enums import Column, Prefix
from typing import Tuple, Any, List
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

__PREDICTED_COUNTRY_RISK_COLUMN = "predicted_country_risk"


def scale_dataset(dataframe: pd.DataFrame, oversample: bool = False) \
        -> Tuple[np.ndarray[Any, np.dtype], np.ndarray, np.ndarray]:
    """
    Scales a given dataset.
    :param dataframe: pandas.Dataframe, A dataframe to sale.
    :param oversample: bool, A toggle whether to oversample or not.
    :return: Tuple[np.ndarray[Any, np.dtype], np.ndarray, np.ndarray], A tuple containing the scaled dataset,
    features and labels
    """
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


def get_distribution(
        test_df: pd.DataFrame,
        y_pred: np.array | List | pd.Series) -> pd.DataFrame:
    """
    Gets a dataframe containing the differences between actual and predicted country risks.
    :param test_df: pandas.Dataframe, The test dataframe, with respect to y_pred.
    :param y_pred: np.array | List | pandas.Series, The predictions of a given model.
    :return: pandas.Dataframe, A dataframe containing the differences between actual and predicted country risks.
    """
    result = test_df.copy()
    result[__PREDICTED_COUNTRY_RISK_COLUMN] = y_pred

    distribution = result.groupby([Column.COUNTRY_RISK, __PREDICTED_COUNTRY_RISK_COLUMN]).size().reset_index().rename(
        columns={0: 'count'})
    distribution["difference"] = distribution[__PREDICTED_COUNTRY_RISK_COLUMN] - distribution[Column.COUNTRY_RISK]
    return distribution


def __get_patch_xy(patch) -> Tuple[float, float]:
    """
    Gets the x y coordinates of a given patch.
    :param patch: matplotlib.patches.Rectangle, A plt.barplot patch.
    :return: Tuple[float, float], A tuple containing the x y coordinates.
    """
    x = patch.get_x() + (patch.get_width() / 2.2)
    y = patch.get_height()
    return x, y


def plot_distribution(distribution: pd.DataFrame) -> None:
    """
    Plots a given distribution.
    :param distribution: pandas.Dataframe, A dataframe containing the differences between actual and predicted country
    risks.
    """
    import matplotlib.patches as mpatches

    correct = distribution["difference"] == 0

    # Plot the 'correct' bar; difference == 0
    correct_y = sum(distribution[correct]["count"])
    correct_container = plt.bar(0, correct_y, color="green")
    plt.annotate(correct_y, __get_patch_xy(correct_container.patches[0]), textcoords="offset points", xytext=(3, 3),
                 ha="center")

    # Plot the 'semi-correct' bars; difference == -1 and 1
    semi_correct_x = [-1, 1]
    semi_correct_y = [
        sum(distribution[distribution["difference"] == x]["count"])
        for x in semi_correct_x]
    semi_correct_container = plt.bar(semi_correct_x, semi_correct_y, color="orange")
    for y, patch in zip(semi_correct_y, semi_correct_container.patches):
        plt.annotate(y, __get_patch_xy(patch), textcoords="offset points", xytext=(3, 3), ha="center")

    # Plot the 'incorrect' bars; difference == else
    incorrect_x = list(distribution["difference"].unique())
    for x in range(-1, 2, 1):
        incorrect_x.remove(x)

    incorrect_y = [
        sum(distribution[distribution["difference"] == x]["count"])
        for x in incorrect_x]
    incorrect_container = plt.bar(incorrect_x, incorrect_y, color="red")
    for y, patch in zip(incorrect_y, incorrect_container.patches):
        plt.annotate(y, __get_patch_xy(patch), textcoords="offset points", xytext=(3, 3), ha="center")

    plt.xlabel("difference")
    plt.ylabel("count")
    plt.xticks(np.arange(min(distribution["difference"]), max(distribution["difference"]) + 1, 1))
    plt.grid(axis="y")
    plt.legend(handles=[
        mpatches.Patch(color='green', label='correct'),
        mpatches.Patch(color='orange', label='semi-correct'),
        mpatches.Patch(color='red', label='incorrect')
    ])
    plt.show()


def map_labels_to_names(series: pd.Series) -> pd.Series:
    """
    Maps a series of labels/classes to their respective names.
    :param series: pandas.Dataframe, A series containing labels/classes.
    :return: pandas.Dataframe, A series containing the names of the given labels/classes.
    """
    from configs.enums import RISKCLASSIFICATIONS

    new_series = series.copy()
    for key, classification in RISKCLASSIFICATIONS.get_attributes().items():
        bool_map = series == classification.value
        new_series.loc[bool_map] = classification.name

    return new_series


def output_incorrectly_predicted_xlsx(
        test_df: pd.DataFrame,
        y_pred: np.array | List | pd.Series,
        model_name: str = "some-model") -> pd.DataFrame:
    """
    Outputs a xlsx file containing the incorrectly predicted rows + additional data from the merged dataset.
    :param test_df: pandas.Dataframe, The test dataframe, with respect to y_pred.
    :param y_pred: np.array | List | pandas.Series, The predictions of a given model.
    :param model_name: str, A name that was given to the model.
    """
    import os
    from configs.data import MERGED_DATASET_PATH, OUT_PATH, VERSION

    result = test_df.copy()
    result[__PREDICTED_COUNTRY_RISK_COLUMN] = y_pred

    incorrectly_predicted = result[result[Column.COUNTRY_RISK] != result[__PREDICTED_COUNTRY_RISK_COLUMN]]

    merged_df = pd.read_excel(MERGED_DATASET_PATH)
    incorrectly_predicted_df = merged_df.iloc[incorrectly_predicted.index,]

    incorrectly_predicted_df["country_risk"] = map_labels_to_names(incorrectly_predicted[Column.COUNTRY_RISK])
    incorrectly_predicted_df[__PREDICTED_COUNTRY_RISK_COLUMN] = map_labels_to_names(incorrectly_predicted[__PREDICTED_COUNTRY_RISK_COLUMN])

    norm_risk = Prefix.NORM + Column.RISK
    cols = [Column.YEAR, Column.COUNTRY] + list(incorrectly_predicted.columns) + [norm_risk]
    incorrectly_predicted_df = incorrectly_predicted_df[cols]
    incorrectly_predicted_df.to_excel(
        os.path.join(OUT_PATH, f"{model_name}-incorrectly-predicted-V.{VERSION}.xlsx"),
        index=False,
        sheet_name="Data")

    return incorrectly_predicted_df
