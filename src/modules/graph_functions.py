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
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches
import configs.plotting as cf


@dataclass
class Axes:
    """An error handling structure, which returns an Axes object if the given parameters were valid."""
    x: pd.Series
    y: pd.Series | None = None
    z: pd.Series | None = None

    @staticmethod
    def from_dict(data: dict) -> Axes:
        """
        Creates an Axes object from a given dict.
        :param data: dict, Containing the x, y and z keys.
        :return: Axes, An Axes object.
        """
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
        """
        :param x: str | pd.Series, A column name or pandas.Series of what should to be plotted.
        :param y: str | pd.Series | None, A column name or pandas.Series (or undefined) of what should to be plotted. .
        :param z: str | pd.Series | None, A column name or pandas.Series (or undefined) of what should to be plotted. .
        :param df: pd.DataFrame | None, A pandas.Series (or undefined) of what should to be plotted. .
        :param y_required: bool = False, A toggle which specifies if y is required.
        :param z_required: bool = False, A toggle which specifies if z is required.
        :return: Axes, An Axes object.
        """
        func_args = locals()
        func_args = {k: v for k, v in func_args.items() if v is not None and k in ("x", "y", "z")}

        if y_required and func_args["y"] is None:
            raise Exception("Argument 'y' is required (cannot be None)")

        if z_required and func_args["z"] is None:
            raise Exception("Argument 'z' is required (cannot be None)")

        if isinstance(x, pd.Series):

            is_pd_ser = [isinstance(v, pd.Series) for k, v in func_args.items()]

            if False in is_pd_ser:
                raise Exception("'x' (, 'y' and 'z') should be either a str or pd.Series and have the same type.")
            if df is not None:
                print("Argument 'df' is unnecessary thus ignored.")

            return Axes.from_dict(func_args)

        is_str_ser = [isinstance(x, str) for k, v in func_args.items()]
        if False in is_str_ser:
            raise Exception("'x' (, 'y' and 'z') should be either a str or pd.Series and have the same type.")

        if df is None:
            raise Exception("'df' should not be None.")

        # v: str = Column name
        return Axes.from_dict({k: df[v] for k, v in func_args.items()})


def _set_labels_plt(
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    """
    Sets the matplotlib labels.
    :param x_label: str | None, The x label.
    :param y_label: str | None, The y label.
    """
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)


def _set_labels_ax3d(
        ax: Axes3D,
        x_label: str | None = None,
        y_label: str | None = None,
        z_label: str | None = None) -> None:
    """
   Sets the Axes3D labels.
   :param x_label: str | None, The value of the x label.
   :param y_label: str | None, The value of the y label.
   :param z_label: str | None, The value of the z label.
   """
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)


def _add_corr_r2_legend(r_sqr: float, r_value: float) -> None:
    """
    Adds a legend containing the r-squared and Pears. Coeff values.
    :param r_sqr: float, R-squared.
    :param r_value: float, R-value | Pears. Coeff.
    """
    r_sqr_patch = mpatches.Patch(label=f"R-squared: {round(r_sqr, 3)}", color="none")
    coeff_patch = mpatches.Patch(label=f"Pear. Coeff: {round(r_value, 3)}", color="none")
    legend = plt.legend(handles=[r_sqr_patch, coeff_patch])
    frame = legend.get_frame()
    frame.set_color(cf.LEGEND_FRAME_COLOR)

    # shift = max([t.get_window_extent().width for t in legend.get_texts()])
    # for t in legend.get_texts():
    #     t.set_horizontalalignment("right")
    #     t.set_position((shift, 0))


def plot_linear(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    """
    Plots a linear regression.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param y: str | pd.Series, Either the column name or the column for the y value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    :param y_label: str | None, The value of the y label.
    """
    axes = Axes.create(df=df, x=x, y=y, y_required=True)
    cf.set_styling()

    sns.regplot(x=axes.x, y=axes.y, line_kws={"color": "red"})

    slope, intercept, r_value, p_value, std_err = sc.linregress(x, y)
    r_sqr = r_value**2
    _add_corr_r2_legend(r_sqr=r_sqr, r_value=r_value)

    _set_labels_plt(x_label, y_label)
    plt.tight_layout()
    plt.show()


def plot_exponential(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    """
    Plots an exponential regression.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param y: str | pd.Series, Either the column name or the column for the y value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    :param y_label: str | None, The value of the y label.
    """
    axes = Axes.create(df=df, x=x, y=y, y_required=True)
    cf.set_styling()

    plt.scatter(x=axes.x, y=axes.y)
    plt.plot(axes.x, [math.pow(10, v) for v in axes.x], color="red")

    _set_labels_plt(x_label, y_label)
    plt.tight_layout()
    plt.show()


def plot_logarithmic(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    """
    Plots a logarithmic regression.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param y: str | pd.Series, Either the column name or the column for the y value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    :param y_label: str | None, The value of the y label.
    """
    axes = Axes.create(df=df, x=x, y=y, y_required=True)
    cf.set_styling()

    plt.scatter(axes.x, axes.y)
    axes.x = list(filter(lambda v: v > 0, axes.x))
    plt.plot(axes.x, [math.log(v) for v in axes.x], color="red")

    _set_labels_plt(x_label, y_label)
    plt.tight_layout()
    plt.show()


def plot_polynomial(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    """
    Plots a polynomial regression.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param y: str | pd.Series, Either the column name or the column for the y value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    :param y_label: str | None, The value of the y label.
    """
    axes = Axes.create(df=df, x=x, y=y, y_required=True)
    cf.set_styling()

    plt.scatter(axes.x, axes.y)

    model = np.poly1d(np.polyfit(x, y, 3))
    line = np.linspace(np.nanmin(x), np.nanmax(x), x.shape[0])
    plt.plot(line, model(line), color="red", label="test")

    patches = [
        mpatches.Patch(label=f"Coeff-{idx}: {round(coeff, 3)}", color="none")
        for idx, coeff in enumerate(model.coefficients)]
    r_sqr = round(r2_score(y, model(x)), 3)
    r_sqr_patch = mpatches.Patch(label=f"R-squared: {r_sqr}", color="none")
    patches.insert(0, r_sqr_patch)
    legend = plt.legend(handles=patches)
    frame = legend.get_frame()
    frame.set_color(cf.LEGEND_FRAME_COLOR)

    _set_labels_plt(x_label, y_label)
    plt.tight_layout()
    plt.show()


def plot_normal_distribution(
        x: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None) -> None:
    """
    Plots a normal distribution.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    """
    axes = Axes.create(df=df, x=x)
    cf.set_styling()

    mu = x.mean()
    sd = x.std()
    plt.plot(x, sc.norm.pdf(axes.x, mu, sd), label=f"μ: {mu:.2f}, σ: {sd:.2f}")

    _set_labels_plt(x_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hist(
        x: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None) -> None:
    """
    Plots a histogram.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    """
    axes = Axes.create(df=df, x=x)
    cf.set_styling()

    mu = x.mean()
    sd = x.std()
    sns.histplot(axes.x, label=f"μ: {mu:.2f}, σ: {sd:.2f}")

    _set_labels_plt(x_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bar(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None) -> None:
    """
    Plots a bar plot.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param y: str | pd.Series, Either the column name or the column for the y value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    """
    axes = Axes.create(df=df, x=x, y=y)
    cf.set_styling()

    sns.barplot(x=axes.x, y=axes.y)

    _set_labels_plt(x_label)
    plt.tight_layout()
    plt.show()


def plot_3d_scatter(
        x: str | pd.Series,
        y: str | pd.Series,
        z: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        z_label: str | None = None) -> None:
    """
    Plots a 3D scatter plot.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param y: str | pd.Series, Either the column name or the column for the y value.
    :param z: str | pd.Series, Either the column name or the column for the z value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    :param y_label: str | None, The value of the y label.
    :param z_label: str | None, The value of the z label.
    """
    axes = Axes.create(df=df, x=x, y=y, z=z, y_required=True, z_required=True)
    cf.set_styling()

    fig = plt.figure(figsize=(9, 6))
    ax = Axes3D(fig)
    fig.add_axes(ax)

    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    scat = ax.scatter(axes.x, axes.y, axes.z, c=axes.y, marker='o', cmap=cmap, alpha=1)

    _set_labels_ax3d(ax, x_label, y_label, z_label)
    plt.legend(*scat.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()


def plot_pairs(df: pd.DataFrame) -> None:
    """
    Plots a pair plot.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    """
    sns.pairplot(df)
    cf.set_styling()
    plt.show()


def plot_kde(
        x: str | pd.Series,
        y: str | pd.Series,
        df: pd.DataFrame | None = None,
        x_label: str | None = None,
        y_label: str | None = None) -> None:
    """
    Plots a KDE plot.
    :param x: str | pd.Series, Either the column name or the column for the x value.
    :param y: str | pd.Series, Either the column name or the column for the y value.
    :param df: None | pd.Series, Either the dataframe which will be plotted or undefined.
    :param x_label: str | None, The value of the x label.
    :param y_label: str | None, The value of the y label.
    """
    axes = Axes.create(df=df, x=x, y=y, y_required=True)
    cf.set_styling()

    slope, intercept, r_value, p_value, std_err = sc.linregress(x, y)
    r_sqr = r_value**2
    _add_corr_r2_legend(r_sqr=r_sqr, r_value=r_value)

    sns.regplot(x=axes.x, y=axes.y, line_kws={"color": "black"}, scatter=False)
    sns.kdeplot(x=axes.x, y=axes.y, cmap="crest", fill=True, bw_adjust=0.5)
    # Also a cool colormap: "rocket"

    _set_labels_plt(x_label, y_label)
    plt.tight_layout()
    plt.show()

