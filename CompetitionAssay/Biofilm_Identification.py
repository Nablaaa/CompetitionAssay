"""
Author: Eric Schmidt

In this script the used area of the cells is calculated to compare
which cell type spreads more.

Results seem to make sense with respect to images
"""

import os
from skimage.io import imread
import numpy as np
import helpers
import polars as pl
from matplotlib import pyplot as plt

# define base directory
base_dir = "dataset_competition_assays/"
competition = "competition_2_WTmScarlet_dwspFmNeonGreen/"
files_are_in = "TW_growth/"


# output directory
output_dir = base_dir + competition + files_are_in + "Competition_results/"

# create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get all the intensity files in the directory
WT_files, Mutant_files = helpers.GetTranswellData(base_dir, competition, files_are_in)
WT_binary, Mutant_binary = helpers.GetTranswellData(
    base_dir, competition, files_are_in + "segmentation/"
)


area_covered_WT = []
area_covered_Mutant = []

all_single_areas_WT = []
all_single_areas_Mutant = []

WT_mean_size, WT_std_size = [], []
Mutant_mean_size, Mutant_std_size = [], []

# go through all the files
for WT_file, Mutant_file, WT_binary_file, Mutant_binary_file in zip(
    WT_files, Mutant_files, WT_binary, Mutant_binary
):
    print(WT_file + " is in progress")
    WT_img = imread(base_dir + competition + files_are_in + WT_file)
    Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)
    WT_segmentation = imread(
        base_dir + competition + files_are_in + "segmentation/" + WT_binary_file
    )
    Mutant_segmentation = imread(
        base_dir + competition + files_are_in + "segmentation/" + Mutant_binary_file
    )

    # now pick a certain label
    WT_binary_img = helpers.PickLabel(WT_segmentation, 1)
    Mutant_binary_img = helpers.PickLabel(Mutant_segmentation, 1)

    # get area and intensity of the cells
    WT_area, WT_intensity = helpers.GetSingleCellAreaAndIntensity(WT_binary_img, WT_img)
    Mutant_area, Mutant_intensity = helpers.GetSingleCellAreaAndIntensity(
        Mutant_binary_img, Mutant_img
    )

    max_WT_approx = np.median(np.sort(WT_intensity)[-30:])
    max_Mutant_approx = np.median(np.sort(Mutant_intensity)[-30:])

    # plot intensity vs area for both types
    plt.figure()
    plt.scatter(WT_area, WT_intensity / max_WT_approx, label="WT", color="b", alpha=0.5)
    plt.scatter(
        Mutant_area,
        Mutant_intensity / max_Mutant_approx,
        label="Mutant",
        color="k",
        alpha=0.5,
    )
    plt.xlabel("Area[px]")
    plt.ylabel("Intensity")
    plt.legend()
    plt.savefig(output_dir + WT_file[:-4] + "_intensity_vs_area.png", dpi=500)


print("done")
