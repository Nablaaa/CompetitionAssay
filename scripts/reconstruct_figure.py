"""
Author: Eric Schmidt

This script is used to recontruct the figure from the paper, based
on the analysis performed in the script Area_competition.py
"""

import matplotlib.pyplot as plt

# import seaborn as sns
import numpy as np
import pandas as pd
import os


def Small_cluster_analysis(num1, num2):
    return num1 / np.sum([num1, num2]), num2 / np.sum([num1, num2])


def main():
    print(__doc__)

    objective_pixel_size = 0.55

    # small threshold
    small_threshold = 20

    base_dir = "example_data/competitions/competition_8_WTmScarlet_dwspFmNeonGreen/TW_growth/Competition_results/"

    filenames = os.listdir(base_dir)
    filenames = np.sort([f for f in filenames if f.endswith(".csv")])[:-1]
    print(filenames)

    small_WT = []
    small_Mutant = []

    for f in filenames:
        # load dataframe
        df = pd.read_csv(base_dir + f)

        # get only area columns
        df = df.loc[:, ["WT_single_cell_area[px]", "Mutant_single_cell_area[px]"]]

        # append to results dataframe
        if f == filenames[0]:
            df_results = df
        else:
            df_results = pd.concat([df_results, df], axis=0)

        # filter out small clusters
        df_small_WT = df["WT_single_cell_area[px]"][
            df["WT_single_cell_area[px]"] < small_threshold / (objective_pixel_size**2)
        ]
        df_small_Mutant = df["Mutant_single_cell_area[px]"].dropna()[
            df["Mutant_single_cell_area[px]"].dropna()
            < small_threshold / (objective_pixel_size**2)
        ]

        small_WT.append(len(df_small_WT))
        small_Mutant.append(len(df_small_Mutant))

    small_WT = np.asarray(small_WT)
    small_Mutant = np.asarray(small_Mutant)

    small_cluster_pairs = [Small_cluster_analysis(x, y) for x, y in zip(small_WT, small_Mutant)]

    plt.figure(figsize=(3, 8))
    plt.scatter([0, 1] * len(small_cluster_pairs), small_cluster_pairs)
    plt.ylim([0, 1])
    plt.show()

    # TODO:
    # find out how much area is covered by small clusters compared to the area that the WT is covering overall
    # because this reduces the effect of growth rate differences

    # multiply df with pixel size and save results in new columns with um2
    df_results["WT_single_cell_area[um2]"] = df_results["WT_single_cell_area[px]"] * (
        objective_pixel_size**2
    )
    df_results["Mutant_single_cell_area[um2]"] = df_results["Mutant_single_cell_area[px]"] * (
        objective_pixel_size**2
    )

    fig, ax = plt.subplots()
    ax.ecdf(df_results["WT_single_cell_area[um2]"], label="WT")
    ax.ecdf(df_results["Mutant_single_cell_area[um2]"].dropna(), label="Mutant")

    # plot vlines at the median of the data
    ax.vlines(df_results["WT_single_cell_area[um2]"].median(), 0, 1, linestyle="--", color="C0")
    ax.vlines(
        df_results["Mutant_single_cell_area[um2]"].dropna().median(),
        0,
        1,
        linestyle="--",
        color="C1",
    )

    plt.legend()
    plt.xscale("log")
    plt.xlim(8, 1e3)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
