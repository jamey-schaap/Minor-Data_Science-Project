import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import scipy.stats as sc
from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


def _parse_graph_data(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        z: str | pd.Series = None) -> Tuple[pd.Series, pd.Series] | Tuple[pd.Series, pd.Series, pd.Series]:

    if (type(x) != str and type(x) != pd.Series)\
            or (type(y) != str and type(y) != pd.Series)\
            or (type(z) != str and type(z) != pd.Series and z is not None):
        raise Exception("Arguments x, y, z have to be the same type and have to be either a str or pd.Series")

    if z is None:
        if type(x) == type(y) == str and type(df) == pd.DataFrame:
            return df[x], df[y]
        elif type(x) == type(y) == pd.Series:
            return x, y

    elif type(x) == type(y) == type(z) == str and type(df) == pd.DataFrame:
        return df[x], df[y], df[z]
    elif type(x) == type(y) == type(z) == pd.Series:
        return x, y, z

    raise Exception("Arguments x, y, z have to be the same type")


def _set_labels(
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)


def plot_linear(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None= None) -> None:
    x, y = _parse_graph_data(df=df, x=x, y=y)

    # x = x[x.index.isin(x.dropna().index)]
    # y = y[y.index.isin(y.dropna().index)]
    #
    # plt.scatter(x, y)
    # a, b = np.polyfit(x, y, 1)
    # plt.plot(x, a * x + b)
    #
    # slope, intercept, r_value, p_value, std_err = sc.linregress(x, y)
    # r_squared = r_value**2

    sns.regplot(x=x, y=y, line_kws={"color": "red"})
    # scatter_kws={"color": "black"}, line_kws={"color": "red"}

    _set_labels(x_label, y_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.tight_layout()
    plt.show()


def plot_exponential(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    x, y = _parse_graph_data(df=df, x=x, y=y)

    plt.scatter(x, y)
    plt.plot(x, [math.pow(10, v) for v in x], color="red")

    _set_labels(x_label, y_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.tight_layout()
    plt.show()


def plot_logarithmic(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    x, y = _parse_graph_data(df=df, x=x, y=y)

    plt.scatter(x, y)
    x = list(filter(lambda v: v > 0, x))
    plt.plot(x, [math.log(v) for v in x], color="red")

    _set_labels(x_label, y_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.tight_layout()
    plt.show()


def plot_polynomial(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    x, y = _parse_graph_data(df=df, x=x, y=y)

    plt.scatter(x, y)

    coefficients = [5, -3, 2, -1]  # Coefficients for the polynomial 5x^3 - 3x^2 + 2x - 1
    polynomial = np.poly1d(coefficients)
    plt.plot(x, polynomial(x), color="red")

    _set_labels(x_label, y_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.tight_layout()
    plt.show()


def plot_normal_distribution(
        x: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None) -> None:
    if type(x) != pd.Series and df is None:
        raise Exception("Invalid parameters, either x has to be a pd.Series or df has to be a pd.DataFrame and x has "
                        "to be a string")

    if type(x) == str and type(df) == pd.DataFrame:
        x = df[x]

    mean = x.mean()
    sd = x.std()
    plt.plot(x, sc.norm.pdf(x, mean, sd), label=f"μ: {mean}, σ: {sd}")

    _set_labels(x_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_3d_plot(
        x: str | pd.Series,
        y: str | pd.Series,
        z: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str = None,
        y_label: str = None,
        z_label: str = None) -> None:
    x, y = _parse_graph_data(df=df, x=x, y=y, z=z)

    fig = plt.figure(figsize=(9, 6))
    ax = Axes3D(fig)
    fig.add_axes(ax)

    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    sc = ax.scatter(x, y, z, c=y, marker='o', cmap=cmap, alpha=1)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()
