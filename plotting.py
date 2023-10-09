import pandas as pd
import graph_functions as gf


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv("datasets/MergedDataset-v1.csv")

    print("Plotting...")
    avg_x_ser = df.groupby(df["year"])["durable"].mean()
    avg_y_ser = df.groupby(df["year"])["GDP_rppp_pc"].mean()
    gf.plot_linear(avg_x_ser, avg_y_ser)

    # gf.scatter_3d_plot(df,
    #                 x="durable",
    #                 y="polity2",
    #                 z="GDP_rppp_pc",
    #                 x_label="Years since regime change",
    #                 y_label="Polity 2 score",
    #                 z_label="GDP per capita")

    #          x                 y                 grouped by
    # --------------------------------------------------------------
    # avg | durable     | GDP_rppp_pc     | year
    # avg | polity2     | GDP_rppp_pc     | year
    # avg | GDP_rppp_pc | test            | grouped by year
    # avg | durable     | test            | year
    # avg | polity2     | test            | year

    # gf.plot_linear(df["durable"], df["GDP_rppp_pc"])
    # gf.plot_exponential(df["polity2"], df["GDP_rppp_pc"])
    # gf.plot_logarithmic(df["durable"], df["GDP_rppp_pc"])
    # gf.plot_polynomial(df["polity2"], df["GDP_rppp_pc"])
    # gf.plot_normal_distribution(df["durable"])


if __name__ == '__main__':
    main()
