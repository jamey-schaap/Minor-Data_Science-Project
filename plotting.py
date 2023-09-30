import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils


def plot_2_y_axis(
        df: pd.DataFrame,
        country: str,
        x_col_name: str,
        y1_col_name: str,
        y2_col_name: str,
        x_label: str = "",
        y1_label: str = "",
        y2_label: str = "") -> None:
    x_label = utils.create_label(x_col_name, x_label)
    y1_label = utils.create_label(y1_col_name, y1_label)
    y2_label = utils.create_label(y2_col_name, y2_label)

    x = df[df["country"] == country][x_col_name]
    y1 = df[df["country"] == country][y1_col_name]
    y2 = df[df["country"] == country][y2_col_name]

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color=color)
    # ax1.set_ylim(np.nanmin(df[y1_col_name]), np.nanmax(df[y1_col_name]))
    # ax1.set_ylim(-11, 11)  # Used for polity2 column
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:red"
    ax2.set_ylabel(y2_label, color=color)
    # ax2.set_ylim(np.nanmin(df[y2_col_name]), np.nanmax(df[y2_col_name]))
    # ax2.set_ylim(0, np.nanmax(df[y2_col_name]))
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(country)
    fig.tight_layout()
    plt.show()


def plot_avg_per_year_2_y_axis(
        df: pd.DataFrame,
        y1_col_name: str,
        y2_col_name: str,
        title: str = "",
        y1_label: str = "",
        y2_label: str = "") -> None:
    y1_label = utils.create_label(y1_col_name, y1_label)
    y2_label = utils.create_label(y2_col_name, y2_label)

    fig, ax1 = plt.subplots()

    avg_pol2_ser = df.groupby(df["year"])[y1_col_name].mean()
    color = "tab:blue"
    ax1.set_xlabel("Year")
    ax1.set_ylabel(y1_label, color=color)
    # ax1.set_ylim(np.nanmin(df[y1_col_name]), np.nanmax(df[y1_col_name]))
    ax1.set_ylim(-11, 11)  # Used for polity2 column
    ax1.plot(avg_pol2_ser.index, avg_pol2_ser.values, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    avg_gdp_ser = df.groupby(df["year"])[y2_col_name].mean()
    color = "tab:red"
    ax2.set_ylabel(y2_label, color=color)
    ax2.plot(avg_gdp_ser.index, avg_gdp_ser.values, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    if title != "":
        plt.title(title)

    fig.tight_layout()
    plt.show()


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv("datasets/MergedDataset-v1.csv")

    print("Plotting...")
    # plot_avg_per_year_2_y_axis(
    #     df=df,
    #     title="Avg. Polity 2 score vs constant-dollar GDP",
    #     y1_col_name="polity2",
    #     y1_label="Polity 2 score",
    #     y2_col_name="GDP_rppp",
    #     y2_label="Constant-dollar GDP (billions)")

    # plot_2_y_axis(
    #     df=df,
    #     country="China",
    #     x_col_name="year",
    #     x_label="Year",
    #     y1_col_name="polity2",
    #     y1_label="Polity 2 score",
    #     y2_col_name="GDP_rppp",
    #     y2_label="Constant-dollar GDP (billions)"
    # )

    # plot_2_y_axis(
    #     df=df,
    #     country="Venezuela",
    #     x_col_name="year",
    #     x_label="Year",
    #     y1_col_name="durable",
    #     y1_label="Years since regime change",
    #     y2_col_name="GDP_rppp",
    #     y2_label="Constant-dollar GDP (billions)"
    # )


if __name__ == '__main__':
    main()
