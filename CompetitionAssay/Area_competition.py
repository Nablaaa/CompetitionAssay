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
WT_binary, Mutant_binary = helpers.GetTranswellData(base_dir, competition, files_are_in + "binary/")


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
    # WT_img = imread(base_dir + competition + files_are_in + WT_file)
    # Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)
    WT_segmentation = imread(base_dir + competition + files_are_in + "binary/" + WT_binary_file)
    Mutant_segmentation = imread(
        base_dir + competition + files_are_in + "binary/" + Mutant_binary_file
    )

    # now pick a certain label
    WT_binary_img = helpers.PickLabel(WT_segmentation, 1)
    Mutant_binary_img = helpers.PickLabel(Mutant_segmentation, 1)

    # calculate the area of the cells
    WT_area_per = helpers.GetCoveredAreaPercent(WT_binary_img)
    Mutant_area_per = helpers.GetCoveredAreaPercent(Mutant_binary_img)

    area_covered_WT.append(WT_area_per)
    area_covered_Mutant.append(Mutant_area_per)

    # get area distribution of single cells
    WT_single_cell_area = helpers.GetSingleCellArea(WT_binary_img)
    all_single_areas_WT.append(WT_single_cell_area)
    WT_mean_size.append(np.mean(WT_single_cell_area))
    WT_std_size.append(np.std(WT_single_cell_area))

    Mutant_single_cell_area = helpers.GetSingleCellArea(Mutant_binary_img)
    all_single_areas_Mutant.append(Mutant_single_cell_area)
    Mutant_mean_size.append(np.mean(Mutant_single_cell_area))
    Mutant_std_size.append(np.std(Mutant_single_cell_area))


all_single_areas_WT = np.concatenate([arr.flatten() for arr in all_single_areas_WT])
all_single_areas_Mutant = np.concatenate([arr.flatten() for arr in all_single_areas_Mutant])


helpers.Plot_Area_Histogram(
    all_single_areas_WT, all_single_areas_Mutant, competition, output_dir, visualize=False
)


# convert to numpy
area_covered_WT = np.array(area_covered_WT)
area_covered_Mutant = np.array(area_covered_Mutant)

ratio_WT_Mutant = area_covered_WT / area_covered_Mutant

repetition_name = [WT_binary[i][-12:-4] for i in range(len(ratio_WT_Mutant))]


# create polars df
df = pl.DataFrame(
    {
        "Name": repetition_name,
        "WT[per]": area_covered_WT,
        "Mutan[per]": area_covered_Mutant,
        "ratio[WT/Mutant]": ratio_WT_Mutant,
        "WT_mean_cluster_size[px]": WT_mean_size,
        "WT_std_cluster_size[px]": WT_std_size,
        "Mutant_mean_cluster_size[px]": Mutant_mean_size,
        "Mutant_std_cluster_size[px]": Mutant_std_size,
    }
)

# save the polars df
df.write_csv(output_dir + "area_covered.csv", separator=",")


print("done")
