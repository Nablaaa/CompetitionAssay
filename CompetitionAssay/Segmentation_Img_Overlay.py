"""
Author: Eric Schmidt

This script is used to test the segmentation results
to check, if the segmentation makes sense
For this, the original images are loaded and the
segmentation is overlayed
The results are saved in a folder as png so that one can
easily click through them
"""

import os
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import helpers

# define base directory
base_dir = "dataset_competition_assays/"
competition = "competition_2_WTmScarlet_dwspFmNeonGreen/"
files_are_in = "TW_growth/"

# output directory
output_dir = base_dir + competition + files_are_in + "Segmentation_overlays/"

# create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

WT_files, Mutant_files = helpers.GetTranswellData(base_dir, competition, files_are_in)
WT_binary, Mutant_binary = helpers.GetTranswellData(
    base_dir, competition, files_are_in + "segmentation/"
)

# go through all the files
for WT_file, Mutant_file, WT_binary_file, Mutant_binary_file in zip(
    WT_files, Mutant_files, WT_binary, Mutant_binary
):
    WT_img = imread(base_dir + competition + files_are_in + WT_file)
    Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)

    WT_segmentation = imread(
        base_dir + competition + files_are_in + "segmentation/" + WT_binary_file
    )
    Mutant_segmentation = imread(
        base_dir + competition + files_are_in + "segmentation/" + Mutant_binary_file
    )

    # # overlay the segmentation
    # ax_WT, WT_overlay_fig = helpers.OverlaySegmentation(WT_img, WT_segmentation)
    # WT_overlay_fig.savefig(output_dir + WT_file[:-4] + ".png", dpi=300)

    # ax_Mutant, Mutant_overlay_fig = helpers.OverlaySegmentation(Mutant_img, Mutant_segmentation)
    # Mutant_overlay_fig.savefig(output_dir + Mutant_file[:-4] + ".png", dpi=300)

    ax, merged_plot = helpers.OverlaySegmentation(
        WT_img, WT_segmentation, Mutant_img, Mutant_segmentation
    )
    merged_plot.savefig(output_dir + WT_file[:-4] + "_merged.png", dpi=300)
