import pandas as pd
from modules import graph_functions as gf


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv("datasets/MergedDataset-v1.csv")

    print("Plotting...")
    # avg_x_ser = df.groupby(df["year"])["GDP_rppp_pc"].mean()
    # avg_y_ser = df.groupby(df["year"])["test"].mean()
    # gf.plot_linear(
    #     avg_x_ser,
    #     avg_y_ser,
    #     x_label="avg. GDP per capita (billions)",
    #     y_label="avg. Sum of investment (billions)",
    #     # y_label="avg. Years since regime change",
    #     # y_label="avg. Polity 2 Score",
    # )

    # gf.scatter_3d_plot(df,
    #                 x="durable",
    #                 y="polity2",
    #                 z="GDP_rppp_pc",
    #                 x_label="Years since regime change",
    #                 y_label="Polity 2 score",
    #                 z_label="GDP per capita")

    gf.plot_linear(
        x=df["durable"],
        y=df["GDP_rppp_pc"],
        # y=pd.Series([math.log(v) for v in df["GDP_rppp_pc"]]),
        # x_label="GDP per capita (billions)",
        # x_label="Sum of investment data (billions)",
        # y_label="Years since regime change"
    )

    # gf.plot_exponential(df["polity2"], df["GDP_rppp_pc"])

    # gf.plot_logarithmic(
    #     x=df["GDP_rppp_pc"],
    #     y=df["durable"],
    #     x_label="GDP per capita (billions)",
    #     y_label="Years since regime change"
    # )

    # gf.plot_polynomial(df["durable"], df["GDP_rppp_pc"])

    # gf.plot_normal_distribution(df["GDP_rppp"])


if __name__ == '__main__':
    main()
