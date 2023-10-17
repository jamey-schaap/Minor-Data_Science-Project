from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import scipy.stats as sc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from dataclasses import dataclass


@dataclass
class Axes:
    x: pd.Series
    y: pd.Series | None = None
    z: pd.Series | None = None

    @staticmethod
    def from_dict(data: dict):
        x = data["x"] if "x" in data.keys() else None
        y = data["y"] if "y" in data.keys() else None
        z = data["z"] if "z" in data.keys() else None
        return Axes(x=x, y=y, z=z)

    @staticmethod
    def create(
            x: str | pd.Series,
            y: str | pd.Series | None = None,
            z: str | pd.Series | None = None,
            df: pd.DataFrame | None = None,
            y_required: bool = False,
            z_required: bool = False) -> Axes:
        func_args = locals()
        func_args = {k: v for k, v in func_args.items() if v is not None and k in ("x", "y", "z")}

        if y_required and func_args["y"] is None:
            raise Exception("Argument 'y' is required (cannot be None)")

        if z_required and func_args["z"] is None:
            raise Exception("Argument 'z' is required (cannot be None)")

        if type(x) == pd.Series:

            is_pd_ser = [type(v) == pd.Series for k, v in func_args.items()]

            if False in is_pd_ser:
                raise Exception("'x' (, 'y' and 'z') should be either a str or pd.Series and have the same type.")
            if df is not None:
                print("Argument 'df' is unnecessary thus ignored.")

            return Axes.from_dict(func_args)

        is_str_ser = [type(v) == str for k, v in func_args.items()]
        if False in is_str_ser:
            raise Exception("'x' (, 'y' and 'z') should be either a str or pd.Series and have the same type.")

        if df is None:
            raise Exception("'df' should not be None.")

        # v: str = Column name
        return Axes.from_dict({k: df[v] for k, v in func_args.items()})


def _set_labels_plt(
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)


def _set_labels_ax3d(
        ax: Axes3D,
        x_label: str | None = None,
        y_label: str | None = None,
        z_label: str | None = None) -> None:
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)


def plot_linear(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    axes = Axes.create(df=df, x=x, y=y, y_required=True)

    x = x.dropna()
    y = y.dropna()
    x = x[x.index.isin(y.index)]
    y = y[y.index.isin(x.index)]

    slope, intercept, r_value, p_value, std_err = sc.linregress(x, y)
    r_squared = r_value**2

    sns.regplot(x=axes.x, y=axes.y, line_kws={"color": "red"})
    # scatter_kws={"color": "black"}, line_kws={"color": "red"}

    _set_labels_plt(x_label, y_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.tight_layout()
    plt.show()


def plot_exponential(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    axes = Axes.create(df=df, x=x, y=y, y_required=True)

    plt.scatter(x=axes.x, y=axes.y)
    plt.plot(axes.x, [math.pow(10, v) for v in axes.x], color="red")

    _set_labels_plt(x_label, y_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.tight_layout()
    plt.show()


def plot_logarithmic(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    axes = Axes.create(df=df, x=x, y=y, y_required=True)

    plt.scatter(axes.x, axes.y)
    axes.x = list(filter(lambda v: v > 0, axes.x))
    plt.plot(axes.x, [math.log(v) for v in axes.x], color="red")

    _set_labels_plt(x_label, y_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.tight_layout()
    plt.show()


def plot_polynomial(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    axes = Axes.create(df=df, x=x, y=y, y_required=True)

    plt.scatter(axes.x, axes.y)

    coefficients = [5, -3, 2, -1]  # Coefficients for the polynomial 5x^3 - 3x^2 + 2x - 1
    polynomial = np.poly1d(coefficients)
    plt.plot(axes.x, polynomial(axes.x), color="red")

    _set_labels_plt(x_label, y_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.tight_layout()
    plt.show()


def plot_normal_distribution(
        x: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None) -> None:
    axes = Axes.create(df=df, x=x)

    mean = x.mean()
    sd = x.std()
    plt.plot(x, sc.norm.pdf(axes.x, mean, sd), label=f"μ: {mean}, σ: {sd}")

    _set_labels_plt(x_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_3d_plot(
        x: str | pd.Series,
        y: str | pd.Series,
        z: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        z_label: str | None = None) -> None:
    axes = Axes.create(df=df, x=x, y=y, z=z, y_required=True, z_required=True)

    fig = plt.figure(figsize=(9, 6))
    ax = Axes3D(fig)
    fig.add_axes(ax)

    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    scat = ax.scatter(axes.x, axes.y, axes.z, c=axes.y, marker='o', cmap=cmap, alpha=1)

    _set_labels_ax3d(ax, x_label, y_label, z_label)

    sns.set_style('darkgrid', {"axed.grid": False})
    plt.legend(*scat.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()
