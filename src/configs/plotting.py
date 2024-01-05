import seaborn as sns
import matplotlib.pyplot as plt

LEGEND_FRAME_COLOR = "white"


def set_styling() -> None:
    sns.set_style(
        'darkgrid',
        {"axed.grid": False})

    plt.figure(
        figsize=(16, 9))

    font = {
        "size": 16}

    plt.rc('font', **font)
