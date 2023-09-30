import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_2_y_axis(df: pd.DataFrame, country: str, x_col_name: str, y1_col_name: str, y2_col_name: str):
    x = df[df["country"] == country][x_col_name]
    y1 = df[df["country"] == country][y1_col_name]
    y2 = df[df["country"] == country][y2_col_name]

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel(x_col_name)
    ax1.set_ylabel(y1_col_name, color=color)
    ax1.set_ylim(np.nanmin(df[y1_col_name]), np.nanmax(df[y1_col_name]))
    # ax1.set_ylim(-11, 11)  # Used for polity2 column
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:red"
    ax2.set_ylabel(y2_col_name, color=color)
    ax2.set_ylim(np.nanmin(df[y2_col_name]), np.nanmax(df[y2_col_name]))
    # ax2.set_ylim(0, np.nanmax(df[y2_col_name]))
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.show()


def plot_avg_2_y_axis(df: pd.DataFrame, y1_col_name: str, y2_col_name: str):
    fig, ax = plt.subplots()

    avg_pol2_ser = df.groupby(df["year"])[y1_col_name].mean()
    avg_pol2_ser.plot(
        ax=ax,
        xlabel="year",
        ylabel=y1_col_name,
        ylim=[-11, 11],
        color="blue")

    avg_gdp_ser = df.groupby(df["year"])[y2_col_name].mean()
    avg_gdp_ser.plot(
        ax=ax,
        ylabel=y2_col_name,
        secondary_y=True,
        ylim=[0, np.nanmax(avg_gdp_ser) + 50],
        color="red")

    fig.tight_layout()
    plt.show()
