import pandas as pd
from modules import graph_functions as gf
from configs.enums import Column, Prefix, Description
from configs.data import MERGED_DATASET_PATH
from typing import Callable


def simple_invoke(
        df: pd.DataFrame,
        x: Column | str,
        y: Column | str,
        plot_func: Callable) -> None:
    get_description = lambda s: s.get_description() if type(s) is Column else Description[str.upper(s)]
    plot_func(
        x=df[x],
        y=df[y],
        x_label=get_description(x),
        y_label=get_description(y))


def main() -> None:
    print("Loading dataset...")
    df = pd.read_excel(MERGED_DATASET_PATH)

    print("Plotting...")
    simple_invoke(df, x=Column.DUR, y=Prefix.NORM + Column.RISK, plot_func=gf.plot_kde)


if __name__ == '__main__':
    main()
