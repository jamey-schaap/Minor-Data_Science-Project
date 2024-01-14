import matplotlib.pyplot as plt
import pandas as pd
from modules import graph_functions as gf
from configs.enums import Column, Prefix, Description
from configs.data import MERGED_DATASET_PATH, MACHINE_LEARNING_DATASET_PATH
from typing import Callable


def get_description(column: str | Column) -> Description | str:
    try:
        description = column.get_description() if type(column) is Column else Description[str.upper(column)]
    except KeyError:
        description = column

    return description


def simple_invoke(
        df: pd.DataFrame,
        x: Column | str,
        y: Column | str,
        plot_func: Callable) -> None:
    """
    Simplified graph function invoker.
    :param df: pd.Dataframe, The data to be plotted.
    :param x: Column | str, The x column to plot.
    :param y: Column | str, The y column to plot.
    :param plot_func: Callable, The plot function to call.
    """

    plot_func(
        x=df[x],
        y=df[y],
        x_label=get_description(x),
        y_label=get_description(y))


def main() -> None:
    print("Loading dataset...")
    df = pd.read_excel(MERGED_DATASET_PATH)

    print("Plotting...")
    simple_invoke(df, x=Column.POL2, y=Column.DUR, plot_func=gf.plot_linear)
    # gf.plot_hist(df["norm_risk"], x_label="Risk factor (0..1)")


if __name__ == '__main__':
    main()
